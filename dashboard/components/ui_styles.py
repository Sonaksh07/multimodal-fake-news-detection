import streamlit as st


def apply_custom_glassmorphism():
    """Injects a restrained, high-clarity editorial design language for academic and intelligence forensics."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --canvas: #F5F3EE;
            --surface: #FFFFFF;
            --sidebar-bg: #EAE7DF;
            --sidebar-active: #DFDCD3;
            --border: #D8D5CC;
            --border-subtle: #E2DFD6;
            --text-primary: #252724;
            --text-secondary: #5C6057;
            --text-muted: #7E8278;
            --accent-forest: #34483C;
            --accent-olive: #747A5A;
            --alert-rust: #A65E45;
            --badge-fake-bg: #F9EFEB;
            --badge-fake-border: #D9B8A8;
            --badge-real-bg: #EDF2EE;
            --badge-real-border: #B7C7BB;
            --badge-neutral-bg: #F3F4EE;
            --badge-neutral-border: #C9CCC0;
            --radius-card: 4px;
        }

        /* Base Typography */
        html, body, p, label, input, button {
            font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: var(--text-primary);
        }

        /* Prevent font overrides from breaking Material Symbols icons (fixes keyboard_double glitch) */
        .material-symbols-rounded, [data-testid="stIcon"], [data-testid="stSidebarCollapseButton"] * {
            font-family: 'Material Symbols Rounded', sans-serif !important;
        }

        code, pre, .stCodeBlock, .mono-text {
            font-family: 'IBM Plex Mono', monospace !important;
        }

        /* Canvas Background */
        .stApp {
            background-color: var(--canvas) !important;
            color: var(--text-primary) !important;
        }

        /* Restrained Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: var(--canvas);
        }
        ::-webkit-scrollbar-thumb {
            background: #CCC9BF;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #B3B0A6;
        }

        /* Headings - Editorial and Authoritative */
        h1, h2, h3, h4 {
            font-family: 'IBM Plex Sans', sans-serif !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.015em !important;
        }
        h1 {
            font-weight: 700 !important;
            font-size: 1.75rem !important;
            margin-bottom: 0.35rem !important;
        }
        h2 {
            font-weight: 600 !important;
            font-size: 1.25rem !important;
            margin-top: 1.2rem !important;
        }
        h3 {
            font-weight: 600 !important;
            font-size: 1.05rem !important;
        }

        p, li {
            color: var(--text-secondary);
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--text-muted) !important;
            font-size: 0.85rem !important;
        }

        /* Structured Editorial Cards (Rectangular, 1px Hairline Border, No Glow) */
        .app-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-card);
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(37, 39, 36, 0.04);
        }

        /* Editorial Banner Frame */
        .editorial-banner {
            background: var(--surface);
            border: 1px solid var(--border);
            border-top: 3px solid var(--accent-forest);
            border-radius: var(--radius-card);
            padding: 24px 28px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(37, 39, 36, 0.04);
        }

        .editorial-tag {
            font-size: 0.70rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--accent-olive);
            margin-bottom: 8px;
        }

        .editorial-title {
            font-size: 1.80rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text-primary);
            margin-bottom: 8px;
            line-height: 1.25;
        }

        .editorial-sub {
            font-size: 0.94rem;
            color: var(--text-secondary);
            line-height: 1.55;
            max-width: 900px;
        }

        .kicker {
            text-transform: uppercase;
            font-size: 0.70rem;
            letter-spacing: 0.08em;
            color: var(--accent-olive);
            font-weight: 700;
            margin-bottom: 4px;
        }

        /* Editorial Classification Badges (No Emojis) */
        .metric-badge-real, .metric-badge-fake, .metric-badge-neutral {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 3px;
            font-weight: 600;
            font-size: 0.82rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .metric-badge-real {
            background: var(--badge-real-bg);
            color: var(--accent-forest);
            border: 1px solid var(--badge-real-border);
        }
        .metric-badge-fake {
            background: var(--badge-fake-bg);
            color: var(--alert-rust);
            border: 1px solid var(--badge-fake-border);
        }
        .metric-badge-neutral {
            background: var(--badge-neutral-bg);
            color: var(--accent-olive);
            border: 1px solid var(--badge-neutral-border);
        }

        /* Model Provenance Label */
        .model-tag {
            display: inline-block;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.74rem;
            color: var(--text-muted);
            background: var(--canvas);
            border: 1px solid var(--border);
            border-radius: 3px;
            padding: 2px 7px;
            margin-top: 4px;
        }

        /* Empty State */
        .empty-state {
            border: 1px dashed var(--border);
            border-radius: var(--radius-card);
            padding: 32px 20px;
            text-align: center;
            color: var(--text-muted);
            background: #FAF9F5;
        }

        /* Sidebar Architecture */
        section[data-testid="stSidebar"] {
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] * {
            color: var(--text-primary);
        }

        /* Transform Sidebar Radio Navigation into an Editorial Vertical List */
        div[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 2px;
            display: flex;
            flex-direction: column;
        }

        div[data-testid="stSidebar"] div[role="radiogroup"] > label {
            padding: 8px 12px !important;
            border-radius: 3px !important;
            background: transparent !important;
            border: 1px solid transparent !important;
            transition: background 0.15s ease !important;
            cursor: pointer !important;
            margin: 0 !important;
            width: 100% !important;
        }

        /* Hide the default radio circle */
        div[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }

        /* Nav item text */
        div[data-testid="stSidebar"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            color: var(--text-secondary) !important;
        }

        /* Nav item hover */
        div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background: var(--sidebar-active) !important;
        }
        div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover div[data-testid="stMarkdownContainer"] p {
            color: var(--text-primary) !important;
        }

        /* Nav item active */
        div[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background: var(--sidebar-active) !important;
            border-left: 3px solid var(--accent-forest) !important;
            border-radius: 0 3px 3px 0 !important;
        }
        div[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
            color: var(--accent-forest) !important;
            font-weight: 600 !important;
        }

        /* Metric KPI Blocks */
        div[data-testid="stMetric"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-card) !important;
            padding: 14px 18px !important;
            box-shadow: 0 1px 2px rgba(37, 39, 36, 0.03) !important;
        }
        div[data-testid="stMetricValue"] > div {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1.45rem !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.01em !important;
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 0.70rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.07em !important;
            color: var(--accent-olive) !important;
        }

        /* Buttons - Restrained & Solid */
        .stButton > button {
            border-radius: 3px !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            padding: 8px 18px !important;
            transition: background 0.15s ease !important;
            border: 1px solid var(--border) !important;
            background-color: var(--surface) !important;
            color: var(--text-primary) !important;
        }
        /* Primary Button (Forest Green Box & Crisp White Text) */
        .stButton > button[kind="primary"],
        .stButton > button[kind="primary"] *,
        .stButton > button[data-testid="baseButton-primary"],
        .stButton > button[data-testid="baseButton-primary"] *,
        button[kind="primary"],
        button[kind="primary"] *,
        button[data-testid="baseButton-primary"],
        button[data-testid="baseButton-primary"] *,
        button[data-testid="baseButton-primary"] p,
        button[kind="primary"] p {
            background-color: #34483C !important;
            border: 1px solid #233128 !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }

        /* Primary Button Disabled State */
        .stButton > button[kind="primary"]:disabled,
        .stButton > button[kind="primary"]:disabled *,
        .stButton > button[data-testid="baseButton-primary"]:disabled,
        .stButton > button[data-testid="baseButton-primary"]:disabled *,
        button[kind="primary"]:disabled,
        button[kind="primary"]:disabled *,
        button[data-testid="baseButton-primary"]:disabled,
        button[data-testid="baseButton-primary"]:disabled *,
        button[data-testid="baseButton-primary"]:disabled p,
        button[kind="primary"]:disabled p {
            background-color: #34483C !important;
            border: 1px solid #233128 !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            opacity: 0.85 !important;
            cursor: not-allowed !important;
        }

        /* Primary Button Hover State */
        .stButton > button[kind="primary"]:hover:not(:disabled),
        .stButton > button[kind="primary"]:hover:not(:disabled) *,
        .stButton > button[data-testid="baseButton-primary"]:hover:not(:disabled),
        .stButton > button[data-testid="baseButton-primary"]:hover:not(:disabled) * {
            background-color: #252724 !important;
            border-color: #121312 !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }

        /* Input Controls & Selectboxes */
        input, textarea, div[data-baseweb="select"] > div {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 3px !important;
            color: var(--text-primary) !important;
            font-size: 0.90rem !important;
        }
        input:focus, textarea:focus {
            border-color: var(--accent-forest) !important;
            box-shadow: none !important;
        }

        /* File Upload Dropzone */
        div[data-testid="stFileUploader"] section {
            background-color: var(--surface) !important;
            border: 1px dashed var(--border) !important;
            border-radius: 3px !important;
            padding: 20px !important;
        }
        div[data-testid="stFileUploader"] section:hover {
            border-color: var(--accent-forest) !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border-radius: 3px 3px 0 0 !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            color: var(--text-secondary) !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: var(--accent-forest) !important;
            border-bottom: 2px solid var(--accent-forest) !important;
            font-weight: 600 !important;
        }

        /* DataTables */
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: var(--radius-card);
            background: var(--surface);
        }

        hr {
            border-color: var(--border) !important;
            margin: 22px 0 !important;
        }

        /* Hide Streamlit auto-generated multi-page sidebar list */
        [data-testid="stSidebarNav"], div[data-testid="stSidebarNavItems"] {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)


def render_verdict_badge(label: str) -> str:
    """Returns clean HTML for an editorial verdict badge without emojis."""
    if label in ("FAKE", "Likely Fake"):
        return "<span class='metric-badge-fake'>Likely Fake</span>"
    if label in ("REAL", "Likely Authentic"):
        return "<span class='metric-badge-real'>Likely Authentic</span>"
    if label in ("Uncertain", "UNCERTAIN"):
        return "<span class='metric-badge-neutral'>Uncertain (Borderline)</span>"
    return f"<span class='metric-badge-neutral'>{label}</span>"


def render_model_tag(model_name: str) -> str:
    """Small caption-style tag showing exactly which model/fallback served a score."""
    return f"<div class='model-tag'>Model: {model_name}</div>"


def render_empty_state(icon: str, message: str, sub: str = "") -> None:
    """Restrained, clean empty state container without emojis."""
    sub_html = f"<div style='margin-top:4px;font-size:0.84rem;color:#7E8278;'>{sub}</div>" if sub else ""
    st.markdown(
        f"<div class='empty-state'><div style='font-weight:600;font-size:0.92rem;color:#252724;'>{message}</div>{sub_html}</div>",
        unsafe_allow_html=True,
    )
