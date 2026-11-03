import streamlit as st

from dashboard.state import get_current_analysis
from dashboard.components.ui_styles import render_empty_state


def render():
    st.title("Explainability & Attribution")
    st.caption("SHAP linguistic token attribution, GradCAM visual localization, graph node influence, and fusion "
               "attribution computed dynamically from the active submission.")

    analysis = get_current_analysis()
    if not analysis:
        render_empty_state(
            "", "No active analysis.",
            "Execute an analysis in Live Detection to inspect explainability artifacts.",
        )
        return

    st.markdown(f"### {analysis.title}")

    t1, t2, t3, t4, t5 = st.tabs([
        "Text (SHAP)", "Image (GradCAM)", "Graph Node Influence", "Fusion Attribution", "Evidence & Reasoning",
    ])

    with t1:
        if not analysis.has_text:
            render_empty_state("", "No article text was submitted with this analysis.")
        else:
            df = analysis.shap_explanation.get("explanation_df") if analysis.shap_explanation else None
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.caption("No attributable tokens found for this text.")

    with t2:
        if not analysis.has_image:
            render_empty_state("", "No image was submitted with this analysis.")
        else:
            st.image(analysis.gradcam_overlay, caption=f"GradCAM Overlay — {analysis.title}", use_container_width=True)

    with t3:
        df_node_imp = analysis.graph_node_importance_df
        if df_node_imp is not None and not df_node_imp.empty:
            st.dataframe(df_node_imp, use_container_width=True)
        else:
            st.caption("No graph node data available for this analysis.")

    with t4:
        res = analysis.fusion_result
        contribs = res["contributions"]
        modality_names = {
            "text_contribution": "Text (BERT/TF-IDF)",
            "image_contribution": "Image (ResNet-50/ELA/pHash)",
            "graph_contribution": "Propagation graph (GAT)",
        }
        top_modality = max(contribs, key=contribs.get)
        st.markdown(
            f"The final verdict of **{res['final_label']}** "
            f"(risk score {res['final_score']:.4f}, confidence {res['confidence_percent']:.2f}%) "
            f"was driven most by **{modality_names[top_modality]}**, contributing "
            f"**{contribs[top_modality]:.1f}%** of the fused score."
        )
        if not analysis.has_text:
            st.caption("Text was not submitted, so its score defaulted to a neutral 0.5 in the fusion above.")
        if not analysis.has_image:
            st.caption("An image was not submitted, so its score defaulted to a neutral 0.5 in the fusion above.")

        contrib_df = {
            "Modality": [modality_names[k] for k in contribs],
            "Relative Contribution (%)": list(contribs.values()),
        }
        st.bar_chart(contrib_df, x="Modality", y="Relative Contribution (%)", color="#34483C")

    with t5:
        from models.reasoning.evidence_engine import build_evidence_list, build_reasoning_paragraph

        st.markdown("<div class='kicker'>Analytical Rationale</div>", unsafe_allow_html=True)
        st.write(build_reasoning_paragraph(analysis))

        st.markdown("---")
        st.markdown("<div class='kicker'>Supporting Forensic Evidence</div>", unsafe_allow_html=True)
        for item in build_evidence_list(analysis):
            st.markdown(f"• {item}")

        st.caption(
            "Every item above is derived from this specific submission's actual model outputs "
            "(SHAP tokens, ELA score, propagation bot ratio, fusion contributions) — nothing here "
            "is a fixed template reused across articles."
        )
