"""
Fits the confidence calibrator (see calibration.py) using real per-modality
model inference over the project's labeled demo corpus, so confidence
values are backed by an actual statistical fit rather than a fixed formula.

Called once from both entry points (dashboard/state.py's cached
`get_models()` and backend/main.py's startup), so it only ever runs once
per process and its result is persisted to disk for subsequent runs.
"""
from __future__ import annotations

import pandas as pd
from configs.logging_config import logger


def ensure_calibrator_fitted(models: dict, sample_size: int = 50) -> None:
    fusion = models["fusion"]
    if fusion.calibrator.is_fitted:
        return

    from datasets.dataset_loader import DatasetLoader

    loader = DatasetLoader()
    try:
        df = loader.load_dataset("isot")
        # Balanced sample of real and fake articles from authentic ISOT dataset
        df_real = df[df["label"] == 0].sample(n=min(sample_size // 2, (df["label"] == 0).sum()), random_state=42)
        df_fake = df[df["label"] == 1].sample(n=min(sample_size // 2, (df["label"] == 1).sum()), random_state=42)
        df = pd.concat([df_real, df_fake], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Falling back to demo dataset for calibration: {e}")
        df = loader.load_dataset("demo")
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    raw_scores, labels = [], []
    for _, row in df.iterrows():
        text = str(row.get("text", "") or row.get("title", ""))
        image_path = row.get("image_path")

        p_text = float(models["bert"].predict_proba([text])[0, 1])
        p_image = float(models["resnet"].predict_proba(image_path)[1]) if image_path else 0.5
        graph = models["builder"].build_propagation_graph_for_article(text, risk_score=p_text, num_nodes=20)
        p_graph, _ = models["gat"].predict_graph_probability(graph)

        raw_score = fusion.w_text * p_text + fusion.w_image * p_image + fusion.w_graph * p_graph
        raw_scores.append(raw_score)
        labels.append(int(row["label"]))

    result = fusion.fit_calibrator(raw_scores, labels)
    logger.info(f"Confidence calibrator bootstrap on ISOT validation sample: {result}")
