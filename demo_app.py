"""Subject-scoped fingerprint comparison using the persisted DenseNet archive."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from src.retrieval.pinecone_store import connect
from src.retrieval.subject_comparison import compare_subject
from src.retrieval.ollama_explanation import explain_comparison

PROJECT_ROOT = Path(__file__).resolve().parent
ARCHIVE = PROJECT_ROOT / 'outputs/embeddings/final_densenet/densenet121_arcface__best-2228dbf0'
st.set_page_config(page_title='Fingerprint Subject Comparison', layout='wide')
st.title('Fingerprint Subject Comparison')
st.caption('DenseNet121 + ArcFace | Pinecone | Other images of the selected subject')
st.info('The exact query image is excluded. Other images of the same finger remain eligible. Cosine similarity is a comparison score, not an accuracy percentage.')

@st.cache_data
def load_metadata(folder):
    root = Path(folder)
    metadata = pd.read_csv(root / 'metadata.csv', dtype={'subject_id': str, 'class_id': str}).fillna('')
    manifest = json.loads((root / 'manifest.json').read_text())
    remote = json.loads((root / 'pinecone.json').read_text())
    if not manifest.get('complete') or not remote.get('complete'):
        raise ValueError('The embedding archive and Pinecone upload must be complete')
    if remote['rows'] != len(metadata) or remote['dimension'] != manifest['embedding_dim']:
        raise ValueError('Archive and Pinecone metadata disagree')
    return metadata, manifest, remote

with st.sidebar:
    st.header('Comparison settings')
    split = st.selectbox('Split', ['test', 'val'])
    try:
        metadata, manifest, remote = load_metadata(str(ARCHIVE / split))
    except (OSError, ValueError, KeyError) as error:
        st.error(str(error))
        st.stop()
    subject = st.selectbox('Subject ID', sorted(metadata.subject_id.unique(), key=int))
    subject_rows = metadata[metadata.subject_id == subject]
    finger = st.selectbox('Query finger', sorted(subject_rows.class_id.unique()))
    finger_rows = subject_rows[subject_rows.class_id == finger].set_index('embedding_row', drop=False)
    query_row = st.selectbox('Query image', finger_rows.index.tolist(),
        format_func=lambda row: f"Image {row} | {finger_rows.loc[row, 'alteration']} | {finger_rows.loc[row, 'difficulty']}")
    top_k = st.slider('Closest other images', 1, 10, 5)
    st.subheader('Ollama explanation')
    explain_results = st.checkbox('Explain comparison results', value=True)
    ollama_model = st.text_input('Installed Ollama model', value='llama3.2',
                                disabled=not explain_results)
    st.caption('Uses local Ollama at 127.0.0.1:11434. Start it with ollama serve and install the selected model with ollama pull MODEL.')

selected = finger_rows.loc[query_row]
st.subheader('1. Selected query fingerprint')
st.write(f'**Subject:** {subject} | **Finger:** {finger} | **Image ID:** {query_row}')
image_path = PROJECT_ROOT / str(selected.filepath)
if image_path.exists():
    st.image(str(image_path), caption='Query image - excluded from results', width=180)
with st.expander('Query metadata'):
    st.dataframe(selected.astype(str).to_frame('Value'), width='stretch')

st.subheader('2. Compare other fingerprint images')
st.caption('Search is restricted to this known subject. This view does not measure identification across subjects.')
if st.button('Compare subject fingerprints', type='primary', width='stretch'):
    try:
        with st.spinner('Finding the closest other images in Pinecone...'):
            vectors = np.load(ARCHIVE / split / 'embeddings.npy', mmap_mode='r')
            index = connect(remote['index'], manifest['embedding_dim'])
            matches = compare_subject(index, remote['namespace'], metadata, vectors, query_row, top_k)
    except Exception as error:
        st.error(f'Comparison failed: {error}')
        st.stop()
    if not matches:
        st.warning('No other images are available for this subject.')
    else:
        results = pd.DataFrame(matches)
        st.subheader('3. Closest other image')
        a, b, c = st.columns(3)
        a.metric('Finger', matches[0]['Fingerprint ID'])
        b.metric('Cosine similarity', f"{matches[0]['Cosine Similarity']:.4f}")
        c.metric('Same finger as query', 'Yes' if matches[0]['Same finger'] else 'No')
        st.subheader('4. Ranked comparisons for this subject')
        st.dataframe(results, hide_index=True, width='stretch')
        st.bar_chart(results.assign(Candidate=results['Rank'].astype(str) + ' | image ' + results['Image ID'].astype(str)),
                     x='Candidate', y='Cosine Similarity', sort=False)
        columns = st.columns(min(5, len(matches)))
        for position, match in enumerate(matches):
            row = metadata[metadata.embedding_row == match['Image ID']].iloc[0]
            path = PROJECT_ROOT / str(row.filepath)
            with columns[position % len(columns)]:
                if path.exists():
                    st.image(str(path), caption=f"#{match['Rank']} | {match['Fingerprint ID']} | image {match['Image ID']}", width=160)
        if explain_results:
            st.subheader('5. Explanation of the results')
            st.caption('AI-generated description of the scores and metadata; no image inspection or verified identity decision.')
            try:
                with st.spinner('Ollama is explaining the comparison results...'):
                    explanation = explain_comparison(matches, {
                        'Subject ID': str(subject), 'Fingerprint ID': str(finger),
                        'Image ID': int(query_row), 'Split': split,
                        'Alteration': str(selected.get('alteration', '')),
                        'Difficulty': str(selected.get('difficulty', '')),
                    }, ollama_model)
                st.write(explanation)
            except Exception as error:
                st.warning(f'Ollama explanation unavailable: {error}. Comparison results above remain available. Check that Ollama is running and the selected model is installed.')
st.caption('Self-image matches are excluded. Different images of the same finger can still have high similarity.')
