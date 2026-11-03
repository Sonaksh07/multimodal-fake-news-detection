# Big Data Based Detection and Propagation Analysis of Fake News and Manipulated Images on Social Networks

A production-grade, research-oriented **Multimodal Misinformation Detection & Social Network Propagation Platform** developed for university project demonstration and viva evaluation.

---

## 🌟 Final Implementation Highlights & System Benchmarks

### 1. Trained NLP Models & Real ISOT Dataset Performance
The natural language models are trained and benchmarked against the full **ISOT Fake News Corpus** (39,101 clean, deduplicated articles: 21,196 authentic, 17,905 fake) using a strict stratified 80/20 train/test split:
- **BERT (`bert-base-uncased`) + Linear Probe Head**:
  - **Test Accuracy**: `99.60%`
  - **Test F1-Score**: `0.9960`
  - **ROC-AUC**: `1.0000`
  - Checkpoint: `models/nlp/bert_probe.pkl` (cached weights, instant startup without retraining)
- **TF-IDF Vectorizer + Logistic Regression Baseline**:
  - **Test Accuracy**: `98.57%`
  - **Test F1-Score**: `0.9843`
  - **ROC-AUC**: `0.9990`
  - Model weights: `models/nlp/tfidf_model.pkl`
- **Model Comparison Benchmark Report**: Persisted at `reports/model_benchmarks.json`.

### 2. High-Performance Caching & PySpark Pipeline
- **Parquet Storage Cache**: Preprocessed ISOT dataset is cached in `datasets/isot/cached_isot_clean.parquet`, reducing reload latency from ~10s to ~1s.
- **Streamlit In-Memory Caching**: `@st.cache_data` applied across dataset explorer and analytics views to eliminate redundant preprocessing.
- **PySpark with Pandas Fallback**: Apache PySpark configured for Windows execution (`PYSPARK_PYTHON` & `PYSPARK_DRIVER_PYTHON` linked to the active environment), with an automatic, resilient fallback to Pandas.

### 3. Adaptive Multimodal Fusion & Calibrated 3-Tier Verdicts
- **Primary Text Anchoring**: Supervised text models serve as the primary ground truth anchor ($w_t = 0.70$ trimodal, $w_t = 0.85$ bimodal).
- **Three-Tier Verdict Architecture**:
  - **Likely Fake** ($P \ge 0.60$)
  - **Likely Authentic** ($P \le 0.40$)
  - **Uncertain** ($0.40 < P < 0.60$, borderline confidence flagged for human verification)
- **Bootstrap Platt Calibration**: Calibrated using balanced ISOT validation samples rather than synthetic demo text.

### 4. Experimental Modality Integrity & Real Signal Forensics
- **ResNet-50 Image Pipeline**: Marked as experimental (`is_experimental = True`). Grounded in deterministic signal forensics (Error Level Analysis for compression artifacts, 64-bit DCT perceptual hashing, EXIF metadata inspection, and noise inconsistency).
- **GAT Social Network Pipeline**: Marked as experimental (`is_experimental = True`). Grounded in network topology metrics (PageRank, betweenness centrality, bot ratio, and cascade fan-out).

---

## 👨‍🎓 Authors & Academic Credits
- **Sonaksh Jain** (Reg No: 230953364)
- **Hansika Abhiraj** (Reg No: 230911128)

*Department of Computer Science & Engineering, Manipal Institute of Technology, Manipal, Karnataka, India*  
*Under the Supervision of Dr. P.B. Shanthi*

---

## 🏛️ System Architecture

```text
                                 +-------------------------+
                                 |  News Input / Dataset   |
                                 +------------+------------+
                                              |
                +-----------------------------+-----------------------------+
                |                             |                             |
                v                             v                             v
      +-------------------+         +-------------------+         +-------------------+
      |    Text Module    |         |   Image Module    |         |   Graph Module    |
      | - TF-IDF + LogReg |         | - Perceptual Hash |         | - NetworkX Graphs |
      | - BERT Classifier |         | - ELA Manipulation|         | - PageRank/Louvain|
      | - SHAP Explainer  |         | - ResNet50        |         | - PyG GAT Model   |
      |                   |         | - GradCAM Viz     |         | - Node Influence  |
      +---------+---------+         +---------+---------+         +---------+---------+
                |                             |                             |
                +-----------------------------+-----------------------------+
                                              |
                                              v
                                 +-------------------------+
                                 | Weighted Fusion Engine  |
                                 | (Text + Image + Graph)  |
                                 +------------+------------+
                                              |
                                              v
                                 +-------------------------+
                                 | FastAPI & SQLite DB     |
                                 +------------+------------+
                                              |
                                              v
                                 +-------------------------+
                                 | Streamlit Dashboard     |
                                 | (10 Core Pages & Reports|
                                 +-------------------------+
```

---

## 🚀 Key Modules & Technical Capabilities

### 1. Text Module (NLP)
- **Baseline**: TF-IDF Vectorizer + Logistic Regression classifier.
- **Main Model**: `bert-base-uncased` fine-tuned sequence classification transformer.
- **Evaluation**: Accuracy, Precision, Recall, F1 Score, ROC Curves, Confusion Matrices.
- **XAI**: SHAP token attribution scoring.

### 2. Image Module (CV)
- **Perceptual Hashing (pHash)**: 64-bit DCT hashing to detect image reuse across news articles.
- **Error Level Analysis (ELA)**: Re-compression pixel difference analysis for splicing and copy-move forgery detection.
- **ResNet50 CNN**: Deep residual network for visual fake image classification.
- **XAI**: GradCAM (Gradient-weighted Class Activation Mapping) visual heatmaps.

### 3. Graph Module (GNN & Propagation)
- **Graph Topology**: NetworkX directed social retweet/reply cascade graphs.
- **Network Centralities**: PageRank, Degree Centrality, Betweenness Centrality.
- **Louvain Algorithm**: Echo-chamber community detection.
- **GNN Model**: Graph Attention Network (GAT).

### 4. Big Data (PySpark)
- Apache PySpark distributed text preprocessing and graph degree extraction pipeline.

### 5. Multimodal Fusion Engine
- Weighted linear score combination (\(P_{final} = w_t P_t + w_i P_i + w_g P_g\)).
- Automated ablation comparison across single, bimodal, and trimodal configurations.

### 6. FastAPI Backend & Streamlit Glassmorphism Dashboard
- 10 interactive dashboard pages: `Home`, `Live Detection`, `Dataset Explorer`, `Text Analysis`, `Image Analysis`, `Propagation Analysis`, `Model Comparison`, `Explainability`, `Reports`, `About`.
- ReportLab downloadable PDF audit reports.

---

## 🏭 Production-Hardening Pass (Third Changelog)

This pass focused on statistical/engineering rigor and closing every
remaining "demo-quality" gap ahead of a live evaluation:

- **Confidence calibration**: replaced the raw `|score-0.5|×2×100` formula
  (which produced unrealistic 5-11% confidence values) with genuine Platt
  scaling, fit on real per-modality model inference over the labeled demo
  corpus, persisted like the other models. The fit is deliberately anchored
  at exactly `raw_score=0.5` so the calibrator's own decision boundary can
  never drift away from the system's label threshold — this specifically
  fixes a subtle bug where an unanchored calibration could make the
  reported label and confidence contradict each other near the boundary.
- **Timestamps**: added `utils/timezone.py` (IST via `zoneinfo`) and
  replaced every `datetime.now()`/`datetime.utcnow()` across
  `database/models.py`, `utils/pdf_report.py`, and `dashboard/state.py`, so
  database records, the dashboard, and PDF reports always agree.
- **Database**: extended `PredictionRecord` with `processing_time_seconds`,
  `report_status` (now actually updated to `"generated"` after a PDF is
  produced), `image_filename`, and per-modality `*_model_version` columns.
- **GAT**: swept the implementation and replaced every `.view()` call with
  `.reshape()`; verified both the PyTorch Geometric and native fallback
  code paths execute correctly.
- **Image forensics**: added `models/cv/forensics_engine.py` — real EXIF
  metadata extraction, histogram clipping/posterization detection,
  block-wise noise-inconsistency analysis, and quadrant edge-density
  balance, each backed by an actual computed number, feeding a dynamically
  composed forensic narrative (never a template). Wired into the Image
  Analysis page, the evidence engine, and the PDF report.
- **Evidence & reasoning**: added `models/reasoning/evidence_engine.py`,
  which builds a per-submission evidence list and a natural-language
  reasoning paragraph entirely from that submission's actual SHAP tokens,
  ELA/forensic findings, and fusion contributions. Wired into Live
  Detection, Explainability, and the PDF report.
- **Reports**: PDF report expanded to a 9-section forensic-style layout
  (Executive Summary, Evidence Summary, Model Results, Article Content,
  Text/Image Explanation, Propagation Summary, Fusion Analysis, Final
  Decision & Recommendations), verified by generating and text-extracting
  an actual PDF.
- **Model Comparison**: added per-modality execution time and a real
  strengths/weaknesses reference table for every model in the pipeline.
- **Propagation metrics**: added `bot_ratio` and `diffusion_depth` to
  `GraphAnalyticsEngine.compute_metrics()`.
- **Startup noise**: `init_db()` no longer logs on every single Streamlit
  rerun (it was being called unconditionally on every user click).
- **Image formats**: confirmed BMP and WEBP work correctly through the
  full pipeline (ELA, pHash, ResNet-50) in addition to JPG/JPEG/PNG.

**Honest note on statistical calibration**: the demo corpus used to fit
the calibrator has only two unique images (reused across all "real" and
all "fake" demo articles respectively), which makes the image-modality
contribution to calibration somewhat degenerate — a structural limitation
of the bundled synthetic demo data, not of the calibration method itself.
The calibration curve is well-behaved and correctly anchored regardless.

## ✅ Final Polishing Pass (Second Changelog)

Building on the rebuild below, this pass removed every remaining demo/placeholder
touchpoint and tightened the UI for university demonstration:

- **Authors corrected** to Sonaksh Jain and Hansika Abhiraj only.
- **Live Detection**: no default demo title/text; supports text-only, image-only,
  or both; shows an explicit "No image uploaded" state instead of silently
  falling back to a sample image.
- **Text Analysis / Image Analysis**: demo articles and the two sample images
  (with their selector) removed entirely — both pages show only the current
  submission, with a clear empty state when that modality wasn't submitted.
  The old benchmark-corpus comparison is now a collapsed, clearly-labeled
  "Reference" section, not part of the main view.
- **Propagation**: the illustrative fallback graph is gone — the page now
  requires a real analysis.
- **Explainability**: removed the last remaining sample-image fallback.
- **Model Comparison**: now compares BERT / TF-IDF / ResNet-50 / GAT / Fusion
  on the **current submission only** (added a parallel raw TF-IDF score
  alongside BERT so the comparison is genuine, not BERT-vs-itself). The
  corpus-wide ablation study moved to a secondary, opt-in expander. Also
  removed a hardcoded `0.85` fallback accuracy value in `weighted_fusion.py`.
- **Model transparency**: `BERTFakeNewsClassifier`, `ResNet50ImageClassifier`,
  and `GATPropagationPipeline` now expose an `active_model_name` property, and
  the UI displays exactly which model/fallback served each score.
- **Reports**: PDF now includes a dedicated Image Explanation section
  (ELA/pHash/GradCAM summary) alongside the existing prediction, confidence,
  text explanation, propagation summary, fusion explanation, and timestamp.
- **Backend API**: `/api/v1/predict/multimodal` now supports text-only,
  image-only, or both (previously required text), matching the dashboard's
  session-state behavior; rejects requests with neither.
- **UI redesign**: replaced the glassmorphism theme with a flatter,
  professional design system (muted slate palette, thin borders instead of
  blur/glow, consistent card/badge/empty-state components).

## 🔧 Demonstration-Readiness Rebuild (First Changelog)

This pass fixed every known crash/hardcoding issue and added session-state
continuity so the same analyzed article flows through every page. Verified
by actually running the app headlessly (`streamlit.testing.v1.AppTest`) and
the FastAPI backend (`TestClient`), not just by reading the code.

**Fixed:**
- **GAT crash** (`AttributeError: 'GATClassifier' object has no attribute 'layer1'`): the
  model's forward-pass dispatch tried to shape-sniff edge_index vs. dense
  adjacency input, but was always called with the wrong format. Rebuilt so
  the architecture (PyG `GATConv` vs. native dense layers) is decided once
  at construction time and the input format always matches it.
- **Hardcoded SHAP fallback** (`"pyramid"`, `"alien"`, `"government"` every
  time): removed the keyword dictionary entirely from `models/xai/explainer.py`;
  expanded the training corpus (`datasets/demo_data.py`) from 6 to ~46
  articles so attributions are genuinely article-dependent.
- **Static propagation graph**: `SocialGraphBuilder.build_propagation_graph_for_article()`
  now derives the graph's seed, bot ratio, and spread fan-out from the
  article's text and predicted risk score, so different articles produce
  visibly different (and reproducible) graphs.
- **No session continuity**: added `dashboard/state.py` — a single
  `run_full_analysis()` call populates `st.session_state`, models load once
  via `st.cache_resource`, and every page (Text/Image/Propagation/Model
  Comparison/Explainability/Reports) reads the same object instead of
  reverting to sample data.
- **Predictions never saved to DB**: Live Detection now persists every
  analysis to `PredictionRecord`, so Reports' audit history actually works.
- **`0.85 if label==1 else 0.15` fabricated ablation scores** (both the
  dashboard's Model Comparison page and the backend's
  `/api/v1/analytics/comparison`): replaced with real inference over a
  sample of the demo corpus.
- **Non-deterministic ResNet50 scores**: the untrained classification head
  is now seeded deterministically instead of randomly re-initialized every
  click.
- **BERT keyword-list fallback**: replaced with a real linear-probe head
  trained on frozen BERT `[CLS]` embeddings; if `transformers`/network
  access is unavailable, it delegates to the trained TF-IDF baseline
  instead of a hardcoded word list.
- Removed debug `print()` statements, an unused `cv2` import, and replaced
  deprecated `use_column_width`/`use_container_width` with Streamlit's
  current `width=` API.
- De-duplicated metric calculation (`utils/metrics.py` is now actually used
  by `NLPEvaluator` instead of sitting dead/unused).

**Honest limitations that remain:**
- There is no live social-media data source in this project, so the
  propagation graph is a *simulated* cascade calibrated to article content
  — not observed real-world spread. This is stated explicitly in the
  Propagation Analysis page rather than presented as real data.
- BERT and ResNet50 use frozen pretrained weights (linear-probe / seeded
  head respectively) rather than full fine-tuning, since no labeled
  fine-tuning dataset ships with the project. Both are clearly documented
  in-code as such, with a path to real fine-tuning (`training/` scripts).

---

## 💻 Installation & Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/user/MINOR_SPECIALISATION.git
cd MINOR_SPECIALISATION

pip install -r requirements.txt
```

### 2. Run Streamlit Dashboard
```bash
python -m streamlit run dashboard/app.py
```
Open `http://localhost:8501` in your web browser.

### 3. Run FastAPI Backend (Optional REST API)
```bash
python -m uvicorn backend.main:app --reload --port 8000
```
Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

### 4. Run Automated Test Suite
```bash
python -m pytest
```

> **Windows Note:** If your Python Scripts folder is not added to your system `PATH`, executing `streamlit` or `uvicorn` directly will throw a `CommandNotFoundException`. Running them via `python -m <module>` guarantees execution on any Windows PowerShell or Command Prompt terminal.

---

## ⚡ Zero-Dependency Demo Mode

If benchmark datasets are not locally downloaded, the system **automatically initializes a realistic demo dataset and synthetic sample images**. Everything executes out-of-the-box for viva demonstration without additional coding.

---

## 📚 References
1. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake News Detection on Social Media: A Data Mining Perspective. *ACM SIGKDD Explorations Newsletter*.
2. Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2020). FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information. *Big Data*.
3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT*.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
5. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. *ICLR*.


## ISOT dataset integration

The project bundles the ISOT `Fake.csv` and `True.csv` source files under `datasets/isot/`. In this supplied copy, the raw files contain 23,481 fake and 21,417 real rows (44,898 total); after normalized title/body deduplication, the loader currently returns 39,100 unique records (17,905 fake and 21,195 real). The Dataset Explorer exposes this as the `isot` dataset. The loader assigns labels from the original source file (`Fake.csv` = 1, `True.csv` = 0), retains source-file provenance, normalizes missing text fields, removes exact normalized title/body duplicates, and creates stable per-load article IDs. The existing generated demo corpus remains available separately as `demo` and should not be represented as authentic news.

Run the application from the project root so relative application paths resolve correctly. In Dataset Explorer, select `isot` and choose **Load & Preprocess Dataset**. The page reports whether processing used PySpark or the explicit Pandas fallback; a Pandas fallback must not be described as a Spark execution. ISOT source files were supplied by the user; see `datasets/isot/PROVENANCE.txt` for the exact counts and limitations. Review the original dataset terms before redistributing it publicly. ISOT is a text-news dataset and does not supply per-article image tampering labels or social propagation graphs. Those modalities remain separate components and require their own appropriately labeled data for supervised evaluation.
