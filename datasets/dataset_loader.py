"""Dataset loading and normalization for authentic ISOT and demo records."""
from pathlib import Path
import pandas as pd
from datasets.demo_data import get_demo_df
from configs.logging_config import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class DatasetLoader:
    """Loads ISOT (authentic source-labeled news) or the separate demo corpus."""
    def __init__(self, data_dir: str = "datasets"):
        candidate = Path(data_dir)
        self.data_dir = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _normalize_isot(cls, fake_path: Path, true_path: Path) -> pd.DataFrame:
        cache_path = fake_path.parent / "cached_isot_clean.parquet"
        # Return cached normalized dataset if it exists and is newer than source CSVs
        if cache_path.exists():
            try:
                cache_mtime = cache_path.stat().st_mtime
                if cache_mtime >= fake_path.stat().st_mtime and cache_mtime >= true_path.stat().st_mtime:
                    cached_df = pd.read_parquet(cache_path)
                    if len(cached_df) > 0 and "label" in cached_df.columns:
                        return cached_df
            except Exception as e:
                logger.warning("Failed to read ISOT parquet cache (%s); regenerating from source CSVs.", e)

        frames = []
        for path, label, source_label in ((fake_path, 1, "ISOT Fake.csv"), (true_path, 0, "ISOT True.csv")):
            if not path.exists():
                raise FileNotFoundError(f"Required ISOT file not found: {path}")
            part = pd.read_csv(path, low_memory=False)
            required = {"title", "text", "subject", "date"}
            missing = required - set(part.columns)
            if missing:
                raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
            part = part.copy()
            part["label"] = label  # Preserve the source dataset's binary class convention.
            part["source"] = "ISOT"
            part["source_file"] = source_label
            part["image_path"] = None
            part["title"] = part["title"].fillna("").astype(str)
            part["text"] = part["text"].fillna("").astype(str)
            part["subject"] = part["subject"].fillna("Unknown").astype(str)
            part["date"] = part["date"].fillna("").astype(str)
            part = part[(part["title"].str.strip() != "") | (part["text"].str.strip() != "")]
            frames.append(part)
        df = pd.concat(frames, ignore_index=True)
        # Deduplicate normalized title+body while retaining source-derived labels.
        key = (df["title"].str.lower().str.strip() + " " + df["text"].str.lower().str.strip()).str.replace(r"\s+", " ", regex=True)
        df = df.loc[~key.duplicated()].copy().reset_index(drop=True)
        df["article_id"] = [f"isot_{i:06d}" for i in range(len(df))]

        # Persist cache for fast sub-second loading on subsequent calls
        try:
            df.to_parquet(cache_path, index=False)
            logger.info("Persisted normalized ISOT cache -> %s (%s records)", cache_path, len(df))
        except Exception as e:
            logger.warning("Could not persist ISOT parquet cache: %s", e)

        return df

    def load_dataset(self, dataset_name: str = "demo") -> pd.DataFrame:
        name = dataset_name.lower()
        if name in {"isot", "isot_fake_real", "isot_combined"}:
            df = self._normalize_isot(self.data_dir / "isot" / "Fake.csv", self.data_dir / "isot" / "True.csv")
            logger.info("Loaded ISOT dataset: %s records (%s real, %s fake)", len(df), int((df.label == 0).sum()), int((df.label == 1).sum()))
            return df
        if name == "demo":
            return get_demo_df()
        file_path = self.data_dir / f"{name}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if "label" not in df.columns:
                raise ValueError(f"Dataset {file_path} has no label column")
            return df
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found. Available datasets: {self.get_available_datasets()}")

    def get_available_datasets(self) -> list:
        names = [p.stem for p in self.data_dir.glob("*.csv")]
        if (self.data_dir / "isot" / "Fake.csv").exists() and (self.data_dir / "isot" / "True.csv").exists():
            names.append("isot")
        names.append("demo")
        return sorted(set(names))
