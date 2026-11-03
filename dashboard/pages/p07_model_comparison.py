import streamlit as st
import pandas as pd
import plotly.express as px

from dashboard.state import get_current_analysis
from dashboard.components.ui_styles import render_empty_state


def render():
    st.title("Model Comparison")
    st.caption("Head-to-head comparison of every model in the pipeline — BERT, TF-IDF, ResNet-50, GAT, and "
               "the fused verdict — computed on the current submission only.")

    analysis = get_current_analysis()
    if not analysis:
        render_empty_state(
            "", "No active analysis.",
            "Execute an analysis in Live Detection to compare model outputs for that submission.",
        )
        return

    _STRENGTHS = {
        "BERT (text)": "Captures context, negation, and sarcasm via pretrained language understanding.",
        "TF-IDF + Logistic Regression (text)": "Fast, interpretable, robust on small/clean vocabularies.",
        "ResNet-50 (image)": "Strong general visual feature extraction from a large pretrained backbone.",
        "GAT (propagation)": "Learns which spreader accounts matter most via attention over the graph.",
        "Weighted Fusion (final)": "Combines all signals; degrades gracefully if one modality is missing.",
    }
    _WEAKNESSES = {
        "BERT (text)": "Requires the classification head to be properly trained/fine-tuned to be reliable.",
        "TF-IDF + Logistic Regression (text)": "Ignores word order and context; vocabulary-limited.",
        "ResNet-50 (image)": "Backbone is not fine-tuned on forgery-specific data in this project.",
        "GAT (propagation)": "Graph is a simulated cascade, not observed real-world spread.",
        "Weighted Fusion (final)": "Fixed relative weights, not learned end-to-end from labeled fusion data.",
    }

    def _verdict_from_prob(p):
        if p >= 0.60:
            return "Likely Fake"
        elif p <= 0.40:
            return "Likely Authentic"
        return "Uncertain"

    rows = []
    if analysis.has_text:
        rows.append({"Model": "BERT (text)", "Modality": "Text", "Risk Score": analysis.p_text,
                     "Verdict": _verdict_from_prob(analysis.p_text),
                     "Execution Time (s)": analysis.text_time_seconds, "Active": analysis.text_model_name})
        rows.append({"Model": "TF-IDF + Logistic Regression (text)", "Modality": "Text", "Risk Score": analysis.p_text_tfidf,
                     "Verdict": _verdict_from_prob(analysis.p_text_tfidf),
                     "Execution Time (s)": None, "Active": "Always computed as baseline reference"})
    if analysis.has_image:
        rows.append({"Model": "ResNet-50 (image)", "Modality": "Image", "Risk Score": analysis.p_image,
                     "Verdict": _verdict_from_prob(analysis.p_image),
                     "Execution Time (s)": analysis.image_time_seconds, "Active": analysis.image_model_name})
    rows.append({"Model": "GAT (propagation)", "Modality": "Graph", "Risk Score": analysis.p_graph,
                 "Verdict": _verdict_from_prob(analysis.p_graph),
                 "Execution Time (s)": analysis.graph_time_seconds, "Active": analysis.graph_model_name})
    rows.append({"Model": "Weighted Fusion (final)", "Modality": "Fusion", "Risk Score": analysis.fusion_result["final_score"],
                 "Verdict": analysis.fusion_result["final_label"],
                 "Execution Time (s)": None, "Active": "Adaptive linear combination"})

    for r in rows:
        r["Strengths"] = _STRENGTHS.get(r["Model"], "")
        r["Weaknesses"] = _WEAKNESSES.get(r["Model"], "")

    df = pd.DataFrame(rows)
    df["Risk Score"] = df["Risk Score"].round(4)

    st.subheader(f"Per-model verdicts — {analysis.title}")
    st.dataframe(
        df[["Model", "Modality", "Risk Score", "Verdict", "Execution Time (s)", "Active"]],
        width="stretch", hide_index=True,
    )
    with st.expander("Model strengths & weaknesses", expanded=False):
        st.dataframe(df[["Model", "Strengths", "Weaknesses"]], width="stretch", hide_index=True)
    st.caption(f"Total pipeline processing time for this submission: {analysis.processing_time_seconds:.2f}s")

    st.markdown("---")
    fig = px.bar(
        df, x="Model", y="Risk Score", color="Verdict",
        color_discrete_map={
            "Likely Fake": "#A65E45", "Likely Authentic": "#34483C", "Uncertain": "#747A5A",
            "FAKE": "#A65E45", "REAL": "#34483C"
        },
        title="Comparative Risk Score by Model", text_auto=".3f",
    )
    fig.update_layout(template="plotly_white", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                       font_color="#252724", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    if analysis.has_text:
        agreement = "agree" if (analysis.p_text >= 0.5) == (analysis.p_text_tfidf >= 0.5) else "disagree"
        st.caption(f"BERT and the TF-IDF baseline **{agreement}** on this article's verdict "
                   f"(BERT={analysis.p_text:.3f}, TF-IDF={analysis.p_text_tfidf:.3f}).")

    st.markdown("---")
    # Official ISOT Benchmark Card
    import json
    import os
    benchmark_file = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "model_benchmarks.json")
    if os.path.exists(benchmark_file):
        try:
            with open(benchmark_file, "r", encoding="utf-8") as f:
                b_data = json.load(f)
            st.subheader("Official ISOT Benchmark (Held-Out Test Set)")
            st.caption(f"Evaluated on {b_data.get('total_records', 39101):,} authentic articles (Train: {b_data.get('train_set_size', 31280):,}, Test: {b_data.get('test_set_size', 7821):,}).")
            b_models = b_data.get("models", {})
            b_rows = []
            for k, v in b_models.items():
                b_rows.append({
                    "Model": v.get("model"),
                    "Test Accuracy": f"{v.get('accuracy') * 100:.2f}%",
                    "Precision": f"{v.get('precision') * 100:.2f}%",
                    "Recall": f"{v.get('recall') * 100:.2f}%",
                    "F1 Score": f"{v.get('f1_score') * 100:.2f}%",
                    "ROC-AUC": f"{v.get('roc_auc'):.4f}",
                    "Train Time (s)": f"{v.get('training_time_sec'):.1f}s",
                })
            if b_rows:
                st.dataframe(pd.DataFrame(b_rows), width="stretch", hide_index=True)
        except Exception:
            pass

    st.markdown("---")
    with st.expander("Reference: modality-combination ablation on the demo corpus", expanded=False):
        st.caption(
            "This benchmarks fusion-weight strategies on a labeled demo corpus for reference only — "
            "it is not part of the current submission's result above."
        )
        from datasets.dataset_loader import DatasetLoader
        from dashboard.state import get_models

        sample_size = st.slider("Sample size", min_value=5, max_value=46, value=15, key="ablation_sample_size")
        if st.button("Run ablation on demo corpus"):
            m = get_models()
            loader = DatasetLoader()
            sample_df = loader.load_dataset("demo").sample(n=min(sample_size, 46), random_state=42).reset_index(drop=True)
            samples = []
            with st.spinner("Running text / image / graph models across the sample..."):
                for _, row in sample_df.iterrows():
                    r_text = row["text"]
                    r_image = row.get("image_path")
                    r_p_text = float(m["bert"].predict_proba([r_text])[0, 1])
                    r_p_image = float(m["resnet"].predict_proba(r_image)[1]) if r_image else 0.5
                    r_graph = m["builder"].build_propagation_graph_for_article(r_text, risk_score=r_p_text, num_nodes=20)
                    r_p_graph, _ = m["gat"].predict_graph_probability(r_graph)
                    samples.append({"p_text": r_p_text, "p_image": r_p_image, "p_graph": r_p_graph, "y_true": int(row["label"])})
            st.session_state["_ablation_samples"] = samples

        samples = st.session_state.get("_ablation_samples")
        if samples:
            m = get_models()
            ablation_results = m["fusion"].evaluate_all_combinations(samples)
            records = [{"Combination": name, "Accuracy": r["accuracy"], "n": r["n_samples"]} for name, r in ablation_results.items()]
            st.dataframe(pd.DataFrame(records).sort_values("Accuracy", ascending=False), width="stretch", hide_index=True)
