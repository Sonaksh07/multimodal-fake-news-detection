import streamlit as st
from datasets.dataset_loader import DatasetLoader
from spark.spark_processor import SparkDataProcessor


@st.cache_data(show_spinner=False)
def _load_and_preprocess_cached(selected: str):
    loader = DatasetLoader()
    df = loader.load_dataset(selected)
    processor = SparkDataProcessor()
    try:
        cleaned = processor.preprocess_text_batch(df)
        engine = processor.engine_used
    finally:
        processor.stop()
    return cleaned, engine


def render():
    st.title("Dataset Explorer")
    st.caption("Inspect and analyze the authentic ISOT news benchmark corpus separately from quick demo samples.")

    loader = DatasetLoader()
    available = loader.get_available_datasets()
    default = available.index("isot") if "isot" in available else 0

    c_sel, c_btn = st.columns([3, 1])
    with c_sel:
        selected = st.selectbox("Select Dataset", available, index=default)
    with c_btn:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        load_clicked = st.button("Load & Preprocess", type="primary", use_container_width=True)

    if "dataset_cache" not in st.session_state:
        st.session_state["dataset_cache"] = {}

    if load_clicked:
        try:
            with st.spinner(f"Loading {selected.upper()} records and executing preprocessing pipeline..."):
                cleaned, engine = _load_and_preprocess_cached(selected)
                st.session_state["dataset_cache"][selected] = (cleaned, engine)
        except Exception as exc:
            st.error(f"Dataset loading/preprocessing failed: {exc}")
            st.exception(exc)

    if selected in st.session_state["dataset_cache"]:
        cleaned, engine = st.session_state["dataset_cache"][selected]

        st.success(f"Loaded and preprocessed {len(cleaned):,} records. Processing Engine: **{engine.upper()}**.")

        if "label" in cleaned.columns:
            real = int((cleaned["label"] == 0).sum())
            fake = int((cleaned["label"] == 1).sum())
        else:
            real = fake = 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Articles", f"{len(cleaned):,}")
        c2.metric("Real Articles", f"{real:,}")
        c3.metric("Fake Articles", f"{fake:,}")
        c4.metric("Engine Used", engine.upper())

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        col_chart, col_info = st.columns([2, 1])
        with col_chart:
            st.subheader("Class Distribution")
            if "label" in cleaned.columns:
                counts = cleaned["label"].value_counts().sort_index().rename(index={0: "Authentic / Real", 1: "Fake News"})
                st.bar_chart(counts, color="#34483C")
        with col_info:
            st.subheader("Corpus Provenance")
            if "source" in cleaned.columns:
                sources = sorted(cleaned["source"].dropna().astype(str).unique())
                st.markdown(f"**Identified Sources:** `{', '.join(sources)}`")
            st.markdown(f"**Cleaned Records:** `{len(cleaned):,}`")
            st.markdown(f"**Storage Format:** `Apache Parquet (Fast Caching)`")

        st.markdown("---")
        st.subheader("Data Records Preview (Top 100 Rows)")
        st.dataframe(cleaned.head(100), use_container_width=True)
        st.caption(f"Previewing 100 of {len(cleaned):,} processed records with normalized body and character metrics.")
    else:
        st.info("Click **Load & Preprocess** above to load the selected dataset into memory.")
