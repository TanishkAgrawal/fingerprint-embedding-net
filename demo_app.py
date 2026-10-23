
"""Interactive presentation demo for fingerprint retrieval + RAG."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# -------------------------------------------------
# Paths and configuration
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

INDEX_DIR = (
    PROJECT_ROOT / "outputs" / "retrieval" / "densenet121"
)

QUERIES_CSV = INDEX_DIR / "queries.csv"
THRESHOLD_JSON = INDEX_DIR / "threshold.json"

st.set_page_config(
    page_title="Fingerprint Recognition System",
    page_icon="🔍",
    layout="wide",
)

# -------------------------------------------------
# Application heading
# -------------------------------------------------

st.title("Fingerprint Recognition System")

st.caption(
    "DenseNet121 + ArcFace | FAISS Vector Retrieval | "
    "Llama 3.2 RAG"
)

st.divider()

# -------------------------------------------------
# Verify required files
# -------------------------------------------------

required_files = [
    INDEX_DIR / "index.faiss",
    INDEX_DIR / "gallery.csv",
    QUERIES_CSV,
    THRESHOLD_JSON,
]

missing = [str(p) for p in required_files if not p.exists()]

if missing:
    st.error("Required retrieval files are missing.")
    st.code("\n".join(missing))
    st.stop()

# -------------------------------------------------
# Load query metadata and threshold
# -------------------------------------------------

queries = pd.read_csv(QUERIES_CSV)

with open(THRESHOLD_JSON, "r", encoding="utf-8") as f:
    threshold_info = json.load(f)

threshold = float(threshold_info["threshold"])

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

with st.sidebar:
    st.header("Retrieval Settings")

    query_number = st.number_input(
        "Query fingerprint number",
        min_value=0,
        max_value=len(queries) - 1,
        value=0,
        step=1,
    )

    top_k = st.slider(
        "Number of retrieved matches",
        min_value=1,
        max_value=10,
        value=5,
    )

    use_rag = st.checkbox(
        "Generate Llama 3.2 explanation",
        value=True,
    )

    st.info(
        f"Calibrated threshold: {threshold:.4f}"
    )

# -------------------------------------------------
# Query information
# -------------------------------------------------

st.subheader("1. Selected Query Fingerprint")

selected = queries.iloc[int(query_number)]

left, right = st.columns(2)

with left:
    st.write("**Query number:**", int(query_number))
    if "class_id" in queries.columns:
        st.write("**Known identity:**", selected["class_id"])

with right:
    st.write("**Source:** SOCOFing test split")
    st.write("**Embedding dimension:** 512")

with st.expander("View fingerprint metadata"):
    st.dataframe(
        selected.to_frame(name="Value"),
        use_container_width=True,
    )

st.divider()

# -------------------------------------------------
# Run existing query.py
# -------------------------------------------------

st.subheader("2. Fingerprint Retrieval")

if st.button(
    "Search Fingerprint Database",
    type="primary",
    use_container_width=True,
):

    command = [
        sys.executable,
        "-m",
        "src.retrieval.query",
        "--index-dir",
        str(INDEX_DIR),
        "--query-number",
        str(int(query_number)),
        "--top-k",
        str(top_k),
        "--threshold",
        str(threshold),
    ]

    if use_rag:
        command.extend([
            "--ollama-model",
            "llama3.2",
        ])

    with st.spinner("Generating embeddings and searching FAISS..."):
        try:
            result = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            st.error("Retrieval timed out.")
            st.stop()

    if result.returncode != 0:
        st.error("Retrieval failed.")
        st.code(result.stderr or result.stdout)
        st.stop()

    output = result.stdout

    # ---------------------------------------------
    # Parse ranked candidates
    # ---------------------------------------------

    matches = []

    pattern = (
        r"^\s*(\d+)\s*\|\s*(.*?)\s*\|\s*"
        r"cosine=([-+]?\d*\.?\d+)"
    )

    for line in output.splitlines():
        match = re.match(pattern, line)

        if match:
            matches.append({
                "Rank": int(match.group(1)),
                "Fingerprint ID": match.group(2).strip(),
                "Cosine Similarity": float(match.group(3)),
            })

    if matches:
        results_df = pd.DataFrame(matches)

        best = results_df.iloc[0]
        best_score = float(best["Cosine Similarity"])
        accepted = best_score >= threshold

        st.subheader("3. Best Matching Fingerprint")

        a, b, c = st.columns(3)

        a.metric(
            "Best Match",
            str(best["Fingerprint ID"]),
        )

        b.metric(
            "Cosine Similarity",
            f"{best_score:.4f}",
        )

        c.metric(
            "Matching Threshold",
            f"{threshold:.4f}",
        )

        if accepted:
            st.success(
                "Top candidate is above the calibrated threshold."
            )
        else:
            st.warning(
                "No candidate exceeded the calibrated threshold."
            )

        st.caption(
            "This is an automated candidate decision, "
            "not proof of personal identity."
        )

        st.subheader("4. Top Retrieved Fingerprints")

        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True,
        )

        st.bar_chart(
            results_df.set_index("Fingerprint ID")[
                "Cosine Similarity"
            ]
        )

    # ---------------------------------------------
    # Display RAG explanation
    # ---------------------------------------------

    if "Local RAG explanation:" in output:
        explanation = output.split(
            "Local RAG explanation:", 1
        )[1].strip()

        st.subheader("5. AI-Generated Explanation")
        st.write(explanation)

    with st.expander("View complete terminal output"):
        st.code(output)

st.divider()

st.caption(
    "Fingerprint embeddings: DenseNet121 | "
    "Similarity search: FAISS | "
    "Generation: Ollama Llama 3.2"
)
