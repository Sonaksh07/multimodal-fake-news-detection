# Project Progress & Implementation Handover

## Project Overview
- **Project Title:** Big Data Based Multimodal Fake News & Image Manipulation Detection System
- **Directory:** `MINOR_SPECIALISATION`
- **Environment:** Windows 64-bit, Python 3.14.2, Java 22.0.2 (CPU execution)

---

## Phase Status Summary
- [x] **Phase 1: Environment & Performance** (COMPLETED)
- [x] **Phase 2: Model Reliability** (COMPLETED)
- [x] **Phase 3: Fusion & Dashboard** (COMPLETED)
- [x] **Phase 4: Testing & Final Handover** (COMPLETED)

---

## Phase 4 Completed Work
1. **Automated Test Suite Expansion & Verification:**
   - Authored `tests/test_api.py` containing 7 comprehensive integration tests covering FastAPI root, `/predict/text`, validation errors, `/predict/multimodal`, `/dataset/sample`, `/analytics/comparison`, and `/reports/generate-pdf`.
   - Verified full 16-test suite (`tests/test_all.py` + `tests/test_api.py`) runs directly from repository root and `MINOR_SPECIALISATION/` with 100% pass rate.
2. **Database Anchoring & ORM Modernization:**
   - Anchored SQLite DB path in `database/db.py` to `Path(__file__).resolve().parent / "fakenews_system.db"`, ensuring consistent path resolution regardless of invocation working directory.
   - Removed orphaned root database directory.
   - Upgraded SQLAlchemy model base import in `database/models.py` to `sqlalchemy.orm.declarative_base` to eliminate SQLAlchemy 2.0 deprecation warnings.
3. **Dependency Manifest & Cross-Platform Packaging:**
   - Updated `requirements.txt` to include `pyarrow>=12.0.0` (for high-speed Parquet dataset caching) and `httpx>=0.24.0` (for asynchronous FastAPI testing).
   - Mirrored `requirements.txt`, `pytest.ini`, and `README.md` to workspace root for direct root execution.
4. **Final Backup Creation:**
   - Generated `backup_final.zip` (clean archive excluding `__pycache__`, `.pytest_cache`, logs, and temporary files; includes complete dataset, trained checkpoints, and test suites).

---

## Phase 3 Completed Work
1. **Adaptive Multimodal Fusion Engine:**
   - Redesigned `WeightedMultimodalFusionEngine` (`models/fusion/weighted_fusion.py`) to prevent untrained or noisy modalities from overriding supervised text signals.
   - Text (BERT/TF-IDF) acts as the primary anchor ($w_{\text{text}} = 0.70$, $w_{\text{image}} = 0.20$, $w_{\text{graph}} = 0.10$).
   - Dynamic weight adaptation: Automatically sets $w_{\text{text}} = 0.85, w_{\text{graph}} = 0.15$ when an image is not submitted; sets $w_{\text{image}} = 0.85, w_{\text{graph}} = 0.15$ when text is not submitted.
2. **Three-Tier Categorical Verdicts:**
   - Replaced fragile binary split with 3 validated tiers:
     - **Likely Fake** ($P_{\text{fake}} \ge 0.60$)
     - **Likely Authentic** ($P_{\text{fake}} \le 0.40$)
     - **Uncertain** ($0.40 < P_{\text{fake}} < 0.60$, borderline confidence flagged for human verification)
   - Updated badge renderers (`dashboard/components/ui_styles.py`), PDF generator (`utils/pdf_report.py`), reasoning engine (`models/reasoning/evidence_engine.py`), and Model Comparison charts (`p07_model_comparison.py`).
3. **Calibrator Alignment on Authentic ISOT Data:**
   - Rewrote `models/fusion/calibration_bootstrap.py` to bootstrap calibration from balanced authentic ISOT validation samples rather than the 46 synthetic demo articles.
   - Grounded Platt-scaling calibrator at decision boundary ($raw\_score = 0.50$).
4. **Dashboard Model Comparison Enhancements:**
   - Displayed official ISOT Benchmark comparison card (BERT 99.6% vs TF-IDF 98.6%) directly in `p07_model_comparison.py`.
   - Updated Plotly bar charts with three-tier color indicators (`#e5566d` for Likely Fake, `#2fb673` for Likely Authentic, `#d9a441` for Uncertain).

---

## Phase 2 Completed Work
1. **ISOT NLP Model Training & Benchmarking:**
   - Script authored at `models/nlp/train_isot_nlp.py` implementing strict stratified 80/20 train/test split.
   - **TF-IDF + Logistic Regression:** Trained on 31,280 ISOT articles. Evaluated on 7,821 held-out test articles.
     - **Accuracy:** 98.57%
     - **F1 Score:** 0.9843
     - **ROC-AUC:** 0.9990
     - Serialized to `models/nlp/tfidf_model.pkl` (248 KB).
   - **BERT Contextual Language Model (`bert-base-uncased`) + Linear Probe:**
     - Extracted 768-dim embeddings across stratified ISOT training split.
     - **Accuracy:** 99.60%
     - **F1 Score:** 0.9960
     - **ROC-AUC:** 1.0000
     - Serialized to `models/nlp/bert_probe.pkl` (7.0 KB).
   - Generated and persisted benchmark report: `reports/model_benchmarks.json`.

2. **Experimental Modality Transparency (ResNet-50 & GAT):**
   - Refactored `models/cv/resnet_model.py` and `models/gnn/gat_model.py` to add `is_experimental = True`.
   - Prevented random noise classification heads from acting as genuine models.
   - Grounded image analysis in deterministic, signal-based forensics (Error Level Analysis JPEG re-compression, perceptual hashing, EXIF tags, and noise inconsistency).
   - Grounded graph propagation risk in graph topology metrics (PageRank, bot ratio, and cascade fan-out) while explicitly labeling the neural propagation layer as an experimental simulation cascade.


---

## Phase 1 Completed Work
1. **Dependency & Environment Verification:**
   - Validated Python 3.14.2 compatibility across PyTorch 2.13.0, Torchvision 0.28.0, PyTorch Geometric 2.8.0, Transformers 5.5.4, Scikit-learn 1.8.0, SHAP 0.52.0, FastAPI 0.135.3, and Streamlit 1.60.0.
   - Identified and fixed PySpark Windows subprocess invocation by binding `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` to `sys.executable`. Verified local PySpark session and operations.
   - Preserved automated, robust fallback to Pandas when Spark or Java is unavailable.

2. **Dataset Optimization & Caching:**
   - Retained the clean, normalized ISOT dataset (39,101 unique deduplicated records: 21,196 real, 17,905 fake).
   - Created a Parquet-backed disk cache (`datasets/isot/cached_isot_clean.parquet`) that drops dataset reload times from ~10s to ~1s.
   - Added `@st.cache_data` in `dashboard/pages/p03_dataset_explorer.py` to prevent redundant reloading during dashboard page switches.

3. **Test Infrastructure:**
   - Added `pytest.ini` at both project root and `MINOR_SPECIALISATION/` so tests execute seamlessly out-of-the-box from any directory.
   - Verified that all 9 test suites pass without import errors.

---

## Commands to Run on Another Laptop / Machine
```powershell
# 1. From workspace root:
cd MINOR_SPECIALISATION

# 2. Run unit tests
python -m pytest

# 3. Start Streamlit Dashboard
python -m streamlit run dashboard/app.py

# 4. Start FastAPI Backend (Optional)
python -m uvicorn backend.main:app --reload --port 8000
```
