import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db import init_db
from dashboard.components.ui_styles import apply_custom_glassmorphism
from dashboard.pages import (
    p01_home, p02_live_detection, p03_dataset_explorer,
    p04_text_analysis, p05_image_analysis, p06_propagation_analysis,
    p07_model_comparison, p08_explainability, p09_reports, p10_about
)

st.set_page_config(
    page_title="Multimodal Misinformation Detection Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()
apply_custom_glassmorphism()

# Sidebar Header Branding - Restrained Editorial Style
st.sidebar.markdown("""
<div style="padding: 10px 0 16px 0; border-bottom: 1px solid #D8D5CC; margin-bottom: 16px;">
    <div style="font-size: 0.68rem; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase; color: #747A5A;">
        Research Platform
    </div>
    <div style="font-size: 1.05rem; font-weight: 600; color: #252724; line-height: 1.3; margin-top: 3px;">
        Multimodal Misinformation & Image Forensics
    </div>
    <div style="font-size: 0.76rem; color: #5C6057; margin-top: 4px;">
        Big Data & Neural Network Analysis
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Menu - Completely Clean Text Without Emojis
menu_mapping = {
    "Overview": "Overview",
    "Live Detection": "Live Detection",
    "Dataset Explorer": "Dataset Explorer",
    "Text Analysis": "Text Analysis",
    "Image Forensics": "Image Forensics",
    "Propagation Analysis": "Propagation Analysis",
    "Model Comparison": "Model Comparison",
    "Explainability": "Explainability",
    "Audit Reports": "Audit Reports",
    "About Platform": "About Platform",
}

selected_display = st.sidebar.radio("Navigation", list(menu_mapping.keys()), label_visibility="collapsed")
choice = menu_mapping[selected_display]

st.sidebar.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

from dashboard.state import get_current_analysis, clear_analysis  # noqa: E402

_analysis = get_current_analysis()
if _analysis:
    is_fake = "Fake" in _analysis.fusion_result["final_label"]
    is_real = "Authentic" in _analysis.fusion_result["final_label"]
    badge_bg = "#F9EFEB" if is_fake else ("#EDF2EE" if is_real else "#F3F4EE")
    badge_text = "#A65E45" if is_fake else ("#34483C" if is_real else "#747A5A")
    badge_border = "#D9B8A8" if is_fake else ("#B7C7BB" if is_real else "#C9CCC0")

    st.sidebar.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #D8D5CC; border-radius:3px; padding:12px; margin-bottom:12px;">
        <div style="font-size:0.68rem; font-weight:700; color:#747A5A; text-transform:uppercase; letter-spacing:0.06em;">Active Submission</div>
        <div style="font-size:0.86rem; font-weight:600; color:#252724; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{_analysis.title[:38]}</div>
        <div style="display:inline-block; margin-top:6px; font-size:0.75rem; font-weight:600; color:{badge_text}; background:{badge_bg}; border:1px solid {badge_border}; border-radius:2px; padding:2px 7px;">
            {_analysis.fusion_result['final_label']} ({_analysis.fusion_result['final_score']:.3f})
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.sidebar.button("Clear Active Submission", use_container_width=True):
        clear_analysis()
        st.rerun()
else:
    st.sidebar.markdown("""
    <div style="background:#FAF9F5; border:1px dashed #D8D5CC; border-radius:3px; padding:10px 12px; text-align:center;">
        <div style="font-size:0.76rem; color:#7E8278;">Ready for analysis</div>
    </div>
    """, unsafe_allow_html=True)

# Footer Information
st.sidebar.markdown("""
<div style="margin-top:24px; padding-top:14px; border-top:1px solid #D8D5CC; font-size:0.72rem; color:#7E8278; line-height:1.45;">
    Department of Computer Science &amp; Engineering<br/>
    Manipal Institute of Technology
</div>
""", unsafe_allow_html=True)

# Page Routing
_pages = {
    "Overview": p01_home.render,
    "Live Detection": p02_live_detection.render,
    "Dataset Explorer": p03_dataset_explorer.render,
    "Text Analysis": p04_text_analysis.render,
    "Image Forensics": p05_image_analysis.render,
    "Propagation Analysis": p06_propagation_analysis.render,
    "Model Comparison": p07_model_comparison.render,
    "Explainability": p08_explainability.render,
    "Audit Reports": p09_reports.render,
    "About Platform": p10_about.render,
}
_pages[choice]()
