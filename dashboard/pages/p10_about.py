import streamlit as st


def render():
    st.title("System Documentation")
    st.caption("Platform specifications, theoretical foundations, and technical architecture")

    st.markdown("""
    <div class="app-card">
        <div class="kicker">Project Specification</div>
        <div style="font-size:1.15rem; font-weight:700; color:#252724; margin-top:4px;">
            Big Data Based Detection and Propagation Analysis of Fake News and Manipulated Images on Social Networks
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="app-card">
            <div class="kicker">Academic Attribution</div>
            <div style="font-size:0.95rem; font-weight:600; color:#252724; margin-top:6px;">
                Sonaksh Jain <span style="font-weight:400; color:#5C6057;">(Reg No: 230953364)</span><br/>
                Hansika Abhiraj <span style="font-weight:400; color:#5C6057;">(Reg No: 230911128)</span>
            </div>
            <div style="color:#5C6057; font-size:0.85rem; margin-top:10px; line-height:1.55;">
                Department of Computer Science &amp; Engineering<br/>
                Manipal Institute of Technology, Manipal, Karnataka, India<br/>
                <i>Under the Supervision of Dr. P.B. Shanthi</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="app-card">
            <div class="kicker">Core System Architecture</div>
            <ul style="margin-top:6px; font-size:0.84rem; color:#252724; line-height:1.75; padding-left:18px;">
                <li><b>Text Modality:</b> BERT contextual language model with trained linear probe head; TF-IDF + Logistic Regression baseline.</li>
                <li><b>Visual Modality:</b> Error Level Analysis (ELA) for compression delta; 64-bit DCT perceptual hashing (pHash); ResNet-50 backbone.</li>
                <li><b>Graph Modality:</b> PyTorch Geometric Graph Attention Network (GAT); PageRank, Betweenness, and Louvain modularity algorithms.</li>
                <li><b>Big Data Layer:</b> Apache PySpark distributed batch processor; in-memory Parquet cache; transparent Pandas fallback.</li>
                <li><b>Fusion &amp; Calibration:</b> Adaptive linear combination with empirical Platt-scaled confidence calibration.</li>
                <li><b>Explainability (XAI):</b> Token attribution via SHAP; visual saliency via Grad-CAM; ReportLab archival PDF generator.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-card">
        <div class="kicker">Empirical Model Benchmarks (ISOT Corpus)</div>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:14px; margin-top:8px;">
            <div style="background:#FAF9F5; border:1px solid #D8D5CC; border-radius:3px; padding:12px;">
                <div style="font-size:0.75rem; font-weight:700; color:#747A5A; text-transform:uppercase;">BERT Linear Probe</div>
                <div style="font-size:1.20rem; font-weight:700; color:#252724; margin-top:2px;">99.60% Accuracy</div>
                <div style="font-size:0.80rem; color:#5C6057;">F1: 0.9960 · ROC-AUC: 1.0000</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #D8D5CC; border-radius:3px; padding:12px;">
                <div style="font-size:0.75rem; font-weight:700; color:#747A5A; text-transform:uppercase;">TF-IDF Baseline</div>
                <div style="font-size:1.20rem; font-weight:700; color:#252724; margin-top:2px;">98.57% Accuracy</div>
                <div style="font-size:0.80rem; color:#5C6057;">F1: 0.9843 · ROC-AUC: 0.9990</div>
            </div>
            <div style="background:#FAF9F5; border:1px solid #D8D5CC; border-radius:3px; padding:12px;">
                <div style="font-size:0.75rem; font-weight:700; color:#747A5A; text-transform:uppercase;">Corpus Partition</div>
                <div style="font-size:1.20rem; font-weight:700; color:#252724; margin-top:2px;">39,101 Records</div>
                <div style="font-size:0.80rem; color:#5C6057;">31,280 Train · 7,821 Test (80/20 Stratified)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
