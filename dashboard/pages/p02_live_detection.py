import streamlit as st
import os

from dashboard.state import run_full_analysis, get_current_analysis
from dashboard.components.ui_styles import render_verdict_badge, render_model_tag, render_empty_state


def render():
    st.title("Live Content Verification")
    st.caption(
        "Submit an article's headline, body text, an image, or both. The evaluation results will propagate "
        "consistently across all diagnostic and analytical modules."
    )

    col_input, col_img = st.columns([3, 2])

    with col_input:
        st.markdown("<div class='kicker'>Article Text Ingestion</div>", unsafe_allow_html=True)
        title = st.text_input("Headline / Title (optional)", value="", placeholder="Enter article headline...")
        text = st.text_area(
            "Body Text", value="", height=180,
            placeholder="Paste complete article text for semantic and linguistic inspection...",
        )

    with col_img:
        st.markdown("<div class='kicker'>Visual Media Ingestion</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Accompanying Image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        image_path = None

        if uploaded_file is not None:
            os.makedirs("uploads", exist_ok=True)
            image_path = f"uploads/{uploaded_file.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(image_path, caption="Submitted Media Record", use_container_width=True)
        else:
            render_empty_state("", "No image provided.", "Analysis will be performed on text modality only.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    can_analyze = bool(text.strip()) or (uploaded_file is not None)

    if not can_analyze:
        st.info("Input article text, upload an image, or supply both to initiate analysis.")

    if st.button("Execute Multimodal Analysis", type="primary", disabled=not can_analyze, use_container_width=True):
        with st.spinner("Executing linguistic, visual forensic, and propagation cascade models..."):
            run_full_analysis(
                title=title.strip() or "Untitled Submission",
                text=text.strip(),
                image_path=image_path,
            )
        st.success("Analysis executed successfully. Diagnostics populated across all modules.")
        st.rerun()

    st.markdown("---")

    analysis = get_current_analysis()
    if not analysis:
        render_empty_state("", "Awaiting content submission.", "Provide text or image input above and click Execute Multimodal Analysis.")
        return

    res = analysis.fusion_result
    st.subheader(f"Evaluation Summary — {analysis.title}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_verdict_badge(res["final_label"]), unsafe_allow_html=True)
    with c2:
        st.metric("Fused Risk Index", f"{res['final_score']:.4f}")
    with c3:
        st.metric("Statistical Confidence", f"{res['confidence_percent']:.2f}%")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kicker'>Modality Breakdown</div>", unsafe_allow_html=True)
    u1, u2, u3 = st.columns(3)
    with u1:
        if analysis.has_text:
            st.metric("Text Channel", f"{analysis.p_text:.4f}", f"Weight: {res['weights']['w_text']:.2f}")
            st.markdown(render_model_tag(analysis.text_model_name), unsafe_allow_html=True)
        else:
            st.metric("Text Channel", "Not Submitted")
    with u2:
        if analysis.has_image:
            st.metric("Vision Channel", f"{analysis.p_image:.4f}", f"Weight: {res['weights']['w_image']:.2f}")
            st.markdown(render_model_tag(analysis.image_model_name), unsafe_allow_html=True)
        else:
            st.metric("Vision Channel", "Not Submitted")
    with u3:
        st.metric("Network Diffusion", f"{analysis.p_graph:.4f}", f"Weight: {res['weights']['w_graph']:.2f}")
        st.markdown(render_model_tag(analysis.graph_model_name), unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kicker'>Forensic Rationale</div>", unsafe_allow_html=True)
    from models.reasoning.evidence_engine import build_reasoning_paragraph
    st.write(build_reasoning_paragraph(analysis))

    st.caption(
        "Modality scores default to neutral baseline (0.50) when that input is omitted. "
        "Review dedicated tabs for in-depth explainability, error level analysis, and diffusion topology."
    )
