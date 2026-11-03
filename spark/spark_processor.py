"""Batch preprocessing using Spark when available, with an explicit Pandas fallback."""
import os
import sys
import pandas as pd
from configs.logging_config import logger

# Ensure PySpark workers on Windows invoke the active Python executable
if "PYSPARK_PYTHON" not in os.environ:
    os.environ["PYSPARK_PYTHON"] = sys.executable
if "PYSPARK_DRIVER_PYTHON" not in os.environ:
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, lower, regexp_replace, length, split, size
    SPARK_AVAILABLE = True
except Exception as exc:
    SPARK_AVAILABLE = False
    logger.warning("PySpark import unavailable; Pandas fallback will be used: %s", exc)

class SparkDataProcessor:
    def __init__(self, app_name: str = "FakeNewsBigDataProcessor"):
        self.app_name = app_name
        self.spark = None
        self.engine_used = "pandas"
        if SPARK_AVAILABLE:
            try:
                self.spark = (SparkSession.builder.appName(app_name)
                    .config("spark.driver.memory", "2g")
                    .config("spark.sql.shuffle.partitions", "4")
                    .config("spark.ui.enabled", "false")
                    .config("spark.pyspark.python", sys.executable)
                    .config("spark.pyspark.driver.python", sys.executable)
                    .getOrCreate())
                self.engine_used = "pyspark"
                logger.info("Apache Spark session initialized")
            except Exception as exc:
                logger.warning("Spark initialization failed (%s); using Pandas fallback", exc)
                self.spark = None
                self.engine_used = "pandas"

    def preprocess_text_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError("Cannot preprocess an empty dataset")
        res = df.copy()
        if "text" not in res.columns:
            raise ValueError("Dataset must contain a 'text' column")
        res["text"] = res["text"].fillna("").astype(str)
        if "title" not in res.columns:
            res["title"] = ""
        res["title"] = res["title"].fillna("").astype(str)
        if self.spark is not None:
            try:
                sdf = self.spark.createDataFrame(res)
                out = (sdf.withColumn("cleaned_text", lower(regexp_replace(col("text"), r"[^a-zA-Z0-9\\s]", "")))
                    .withColumn("word_count", size(split(col("cleaned_text"), r"\\s+")))
                    .withColumn("char_count", length(col("text"))))
                return out.toPandas()
            except Exception as exc:
                logger.exception("Spark processing failed; falling back to Pandas: %s", exc)
                self.engine_used = "pandas_fallback"
        res["cleaned_text"] = res["text"].str.lower().str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
        res["word_count"] = res["cleaned_text"].str.split().str.len().fillna(0).astype(int)
        res["char_count"] = res["text"].str.len()
        return res

    def compute_graph_degree_batch(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        if edges_df is None or edges_df.empty:
            return pd.DataFrame(columns=["user_id", "out_degree", "in_degree", "total_degree"])
        if self.spark is not None:
            try:
                edges = self.spark.createDataFrame(edges_df)
                outd = edges.groupBy("source").count().withColumnRenamed("count", "out_degree").withColumnRenamed("source", "user_id")
                ind = edges.groupBy("target").count().withColumnRenamed("count", "in_degree").withColumnRenamed("target", "user_id")
                return outd.join(ind, "user_id", "outer").fillna(0).withColumn("total_degree", col("out_degree") + col("in_degree")).toPandas()
            except Exception as exc:
                logger.warning("Spark graph processing failed: %s", exc)
        src = edges_df["source"].value_counts().rename("out_degree")
        dst = edges_df["target"].value_counts().rename("in_degree")
        out = pd.concat([src, dst], axis=1).fillna(0).reset_index().rename(columns={"index": "user_id"})
        out["total_degree"] = out["out_degree"] + out["in_degree"]
        return out

    def stop(self):
        if self.spark is not None:
            self.spark.stop()
            self.spark = None
