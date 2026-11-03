import streamlit as st


def render():
    # Editorial Header Banner
    st.markdown("""
    <div class="editorial-banner">
        <div class="editorial-tag">Research &amp; Forensics Platform</div>
        <div class="editorial-title">
            Multimodal Misinformation and Image Tampering Detection
        </div>
        <div class="editorial-sub">
            An empirical investigation system integrating transformer-based language representations, 
            digital image forensics, and graph neural network diffusion modeling for comprehensive veracity assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # System Performance & Corpus Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Benchmark Corpus", "39,101", "ISOT Cleaned Records")
    with c2:
        st.metric("BERT Transformer", "99.60%", "F1: 0.9960 | AUC: 1.00")
    with c3:
        st.metric("TF-IDF Baseline", "98.57%", "F1: 0.9843 | AUC: 0.99")
    with c4:
        st.metric("Detection Modalities", "3 Channels", "Text · Vision · Graph")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 4 Architectural Modules (2x2 Grid)
    st.markdown("<div class='kicker'>Architecture Overview</div>", unsafe_allow_html=True)
    st.subheader("Tri-Channel Forensic Pipeline")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="app-card">
            <div style="font-size:1.02rem; font-weight:600; color:#252724; margin-bottom:6px;">
                Natural Language Processing
            </div>
            <p style="font-size:0.88rem; color:#5C6057; margin-bottom:12px;">
                Trained and evaluated on 39,101 clean ISOT records to identify syntactic anomalies, semantic distortion, and linguistic indicators of deceptive reporting.
            </p>
            <div style="font-size:0.83rem; color:#252724; line-height:1.7;">
                • <b>BERT Classifier:</b> 768-dimensional contextual embeddings with trained linear probe head.<br/>
                • <b>TF-IDF Baseline:</b> 5,000 n-gram features paired with L2-regularized logistic regression.<br/>
                • <b>Interpretability:</b> Token-level attribution scoring computed via SHAP.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="app-card">
            <div style="font-size:1.02rem; font-weight:600; color:#252724; margin-bottom:6px;">
                Social Network Propagation Modeling
            </div>
            <p style="font-size:0.88rem; color:#5C6057; margin-bottom:12px;">
                Models social diffusion cascades to analyze dissemination velocity, bot amplification signatures, and echo-chamber clustering.
            </p>
            <div style="font-size:0.83rem; color:#252724; line-height:1.7;">
                • <b>Graph Attention Network (GAT):</b> PyG 2-layer multi-head self-attention over node features.<br/>
                • <b>Structural Metrics:</b> PageRank, Betweenness Centrality, and In/Out degree distribution.<br/>
                • <b>Community Detection:</b> Louvain modularity algorithm for echo-chamber partition.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="app-card">
            <div style="font-size:1.02rem; font-weight:600; color:#252724; margin-bottom:6px;">
                Computer Vision &amp; Digital Forensics
            </div>
            <p style="font-size:0.88rem; color:#5C6057; margin-bottom:12px;">
                Deterministic signal analysis identifying digital manipulation, copy-move artifacts, splicing boundaries, and viral media reuse.
            </p>
            <div style="font-size:0.83rem; color:#252724; line-height:1.7;">
                • <b>Error Level Analysis (ELA):</b> JPEG re-compression rate delta highlighting altered regions.<br/>
                • <b>Perceptual Hashing (pHash):</b> 64-bit discrete cosine transform image fingerprinting.<br/>
                • <b>Deep Vision Backbone:</b> ResNet-50 visual feature representation with Grad-CAM localization.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="app-card">
            <div style="font-size:1.02rem; font-weight:600; color:#252724; margin-bottom:6px;">
                Distributed Big Data &amp; Adaptive Fusion
            </div>
            <p style="font-size:0.88rem; color:#5C6057; margin-bottom:12px;">
                High-throughput distributed data pipelines coupled with calibrated multimodal Bayesian-style decision fusion.
            </p>
            <div style="font-size:0.83rem; color:#252724; line-height:1.7;">
                • <b>Apache PySpark:</b> Distributed text preprocessing and token frequency scaling.<br/>
                • <b>Parquet Cache:</b> Column-oriented disk caching cutting reload time to ~1s.<br/>
                • <b>Adaptive Fusion:</b> Dynamic modality weighting with three-tier calibrated verdicts.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Operational Methodology
    st.markdown("<div class='kicker'>Operational Methodology</div>", unsafe_allow_html=True)
    st.subheader("Forensic Investigation Workflow")

    st.markdown("""
    <div class="app-card">
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap:16px;">
            <div style="background:#FAF9F5; border:1px solid #D8D5CC; border-radius:3px; padding:14px;">
                <div style="font-size:0.75rem; font-weight:700; color:#747A5A; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">01. Ingestion</div>
                <div style="font-size:0.92rem; font-weight:600; color:#252724; margin-bottom:4px;">Content Submission</div>
                <div style="font-size:0.82rem; color:#5C6057;">Submit headline text, article body, and/or accompanying image in Live Detection.</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #D8D5CC; border-radius:3px; padding:14px;">
                <div style="font-size:0.75rem; font-weight:700; color:#747A5A; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">02. Inspection</div>
                <div style="font-size:0.92rem; font-weight:600; color:#252724; margin-bottom:4px;">Unimodal Diagnostics</div>
                <div style="font-size:0.82rem; color:#5C6057;">Inspect isolated ELA heatmaps, pHash signatures, and cascade topologies across dedicated modules.</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #D8D5CC; border-radius:3px; padding:14px;">
                <div style="font-size:0.75rem; font-weight:700; color:#747A5A; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">03. Evaluation</div>
                <div style="font-size:0.92rem; font-weight:600; color:#252724; margin-bottom:4px;">Comparative Analysis</div>
                <div style="font-size:0.82rem; color:#5C6057;">Assess BERT vs TF-IDF vs ResNet-50 vs GAT score distributions in Model Comparison.</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #D8D5CC; border-radius:3px; padding:14px;">
                <div style="font-size:0.75rem; font-weight:700; color:#747A5A; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">04. Verification</div>
                <div style="font-size:0.92rem; font-weight:600; color:#252724; margin-bottom:4px;">Audit &amp; Reporting</div>
                <div style="font-size:0.82rem; color:#5C6057;">Review token-level SHAP attributions and export an archival research PDF in Audit Reports.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
