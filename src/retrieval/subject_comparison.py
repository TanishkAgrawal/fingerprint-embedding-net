"""Subject-scoped comparison of archived images, excluding the exact query."""
import numpy as np


def compare_subject(index, namespace, metadata, vectors, query_row, top_k=5):
    rows = metadata.set_index('embedding_row', drop=False)
    if not rows.index.is_unique or not 1 <= top_k <= 10:
        raise ValueError('Unique image IDs and top-k between 1 and 10 are required')
    query = rows.loc[int(query_row)]
    subject_rows = rows[rows.subject_id.astype(str) == str(query.subject_id)]
    eligible = subject_rows[subject_rows.embedding_row != int(query_row)]
    for column in ('filepath', 'processed_path'):
        if column in rows and str(query[column]):
            paths = eligible[column].astype(str).str.replace('\\', '/', regex=False).str.lower()
            eligible = eligible[paths != str(query[column]).replace('\\', '/').lower()]
    if eligible.empty:
        return []
    response = index.query(namespace=namespace,
        vector=np.asarray(vectors[int(query_row)], dtype=np.float32).tolist(),
        top_k=len(subject_rows),
        filter={'class_id': {'$in': subject_rows.class_id.astype(str).unique().tolist()}},
        include_values=False, include_metadata=False, timeout=30)
    matches = []
    for match in response.matches:
        row_id = int(match.id)
        if row_id not in rows.index:
            raise ValueError('Remote image ID missing from archive metadata')
        candidate = rows.loc[row_id]
        if str(candidate.subject_id) != str(query.subject_id):
            raise ValueError('Remote search returned a different subject')
        if row_id not in eligible.index:
            continue
        matches.append({'Rank': len(matches) + 1, 'Image ID': row_id,
            'Subject ID': str(candidate.subject_id), 'Fingerprint ID': str(candidate.class_id),
            'Cosine Similarity': float(match.score),
            'Same finger': str(candidate.class_id) == str(query.class_id),
            'Alteration': str(candidate.get('alteration', '')),
            'Difficulty': str(candidate.get('difficulty', ''))})
        if len(matches) == top_k:
            break
    return matches
