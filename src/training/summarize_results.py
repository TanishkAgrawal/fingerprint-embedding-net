"""Build the final results table from saved evidence; no training or remote calls."""
import csv
import json
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
rows = []

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

def relative_checkpoint(value):
    value = value.replace('\\', '/')
    return value[value.index('checkpoints/'):] if 'checkpoints/' in value else value

def pct(value):
    return f'{100 * float(value):.4f}%'

def add(method, model, checkpoint, sample, result, source, status='Saved result'):
    checkpoint = relative_checkpoint(checkpoint)
    if not (ROOT / checkpoint).exists():
        status += '; recorded checkpoint path missing'
    rows.append(dict(method=method, model=model, checkpoint=checkpoint, sample=sample,
                     result=result, status=status, source=source))

# A: only these rows share the exact same saved image pairs.
comparison = read('outputs/eer_comparison/summary.json')
with (ROOT / 'outputs/eer_comparison/test_pairs.csv').open(newline='') as stream:
    pairs = list(csv.DictReader(stream))
pair_keys = [(r['row_a'], r['row_b'], r['genuine']) for r in pairs]
for report in comparison:
    assert 'error' not in report, report
    score_path = 'outputs/eer_comparison/' + report['model'] + '_scores.csv'
    with (ROOT / score_path).open(newline='') as stream:
        scores = list(csv.DictReader(stream))
    assert [(r['row_a'], r['row_b'], r['genuine']) for r in scores] == pair_keys
    assert sum(int(r['genuine']) for r in scores) == report['genuine_pairs']
    assert len(scores) - report['genuine_pairs'] == report['impostor_pairs']
    add('A: shared-pair verification', report['model'], report['checkpoint'],
        f"{report['genuine_pairs']:,} genuine + {report['impostor_pairs']:,} impostor pairs; {report['unique_images']:,} images; seed {report['seed']}",
        f"EER {pct(report['eer'])}; AUC {report['auc']:.6f}",
        'outputs/eer_comparison/' + report['model'] + '.json')

# B: preserve the checkpoint string and epoch as recorded; do not rewrite old paths.
for path in sorted((ROOT / 'outputs/results').rglob('report.json')):
    report = read(path.relative_to(ROOT))
    verify, identify = report['verification_overall'], report['identification']
    add('B: historical local full evaluation', f"{report['backbone']} (epoch {report['epoch']})",
        report['checkpoint'],
        f"{verify['n_genuine']:,} genuine + {verify['n_impostor']:,} impostor pairs; {identify['n_gallery']:,} gallery / {identify['n_probe']:,} probes",
        f"EER {pct(verify['eer'])}; AUC {verify['roc_auc']:.6f}; Rank-1 {pct(identify['rank_1'])}; Recall@5 {pct(identify['rank_5'])}",
        path.relative_to(ROOT).as_posix(), 'Historical report; checkpoint bytes not fingerprinted')

# C: saved FAISS-era retrieval evidence must not be presented as current Pinecone accuracy.
for path in sorted((ROOT / 'outputs/retrieval').glob('*/eval_report.json')):
    report = read(path.relative_to(ROOT))
    full = report['evaluation_full']
    add('C: historical enrollment retrieval', path.parent.name, report['checkpoint_read_only'],
        f"{report['gallery_size']:,} enrolled fingers / {full['n_queries']:,} queries",
        f"Rank-1 {pct(full['rank1'])}; Recall@5 {pct(full['recall_at_5'])}; Recall@10 {pct(full['recall_at_10'])}",
        path.relative_to(ROOT).as_posix(), 'Historical report; not a fresh Pinecone evaluation')

# D: verify per-person scores against raw query results and recover checkpoint provenance.
archives = {'densenet121_arcface': 'outputs/embeddings/final_densenet/densenet121_arcface__best-2228dbf0',
            'resnet18': 'outputs/embeddings/resnet18_arcface__best-10ae17bb'}
for model, archive in archives.items():
    base = f'outputs/person_evaluation/{model}'
    report = read(base + '/summary.json')
    with (ROOT / (base + '/query_results.csv')).open(newline='') as stream:
        detail = list(csv.DictReader(stream))
    groups = {}
    for record in detail:
        groups.setdefault((record['split'], record['subject_id']), []).append(record)
    assert len(detail) == report['queries'] and len(groups) == report['people']
    values = []
    for label, position in ((2, 1), (4, 3), (6, 5)):
        measured = sum(sum(r[f'rank{label}_correct'].lower() == 'true' for r in group) / len(group)
                       for group in groups.values()) / len(groups)
        assert abs(measured - report['average_accuracy'][f'rank{label}']) < 1e-10
        values.append(f"candidate position {position}: {pct(measured)}")
    checkpoint = read(archive + '/test/manifest.json')['checkpoint']
    assert relative_checkpoint(read(archive + '/val/manifest.json')['checkpoint']) == relative_checkpoint(checkpoint)
    add('D: Pinecone archive, exact self removed', model, checkpoint,
        f"{report['people']} people / {report['queries']:,} queries; validation + test; seed {report['seed']}",
        '; '.join(values), base + '/summary.json', 'Reconciled with query-level results; checkpoint from archive manifests')

threshold_path = 'outputs/retrieval/densenet121_final/threshold.json'
threshold = read(threshold_path)
add('E: validation threshold calibration', 'densenet121_arcface', threshold['checkpoint'],
    f"{threshold['genuine_pairs']:,} genuine + {threshold['sampled_impostor_pairs']:,} sampled impostor validation pairs",
    f"Threshold {threshold['threshold']:.6f}; FAR {pct(threshold['empirical_val_far'])}; FRR {pct(threshold['empirical_val_frr'])}", threshold_path)
manifest_path = 'outputs/retrieval/densenet121_final/manifest.json'
manifest = read(manifest_path)
assert relative_checkpoint(threshold['checkpoint']) == relative_checkpoint(manifest['checkpoint'])
add('F: final Pinecone enrollment evaluation', 'densenet121_arcface', manifest['checkpoint'],
    f"{manifest['gallery_size']:,} enrolled fingers / {manifest['query_size']:,} intended queries",
    'No completed full evaluation report; accuracy not available', manifest_path, 'Pending')
add('G: current subject-scoped frontend', 'densenet121_arcface', manifest['checkpoint'],
    'User-selected subject and image; exact query and duplicate paths excluded',
    'Interactive similarity ranking; no aggregate accuracy measured', 'src/retrieval/subject_comparison.py', 'Demo only')

intro = '''# Final results comparison

Prepared 29 October 2026 from saved project artifacts. No models were retrained and no remote evaluation was run to build this table.

**Primary model comparison: method A.** All 12 checkpoints use the same 2,000 saved test pairs (1,000 genuine and 1,000 impostor), covering 3,179 images, seed 42. Pair IDs and labels were checked against every model score file. DenseNet121 ArcFace best has the lowest observed EER in this sample (0.0000%, AUC 1.000000). This does not establish zero error on all fingerprints, and the test comparison is retrospective evidence, not an independent estimate after selecting a model on these results.

**Read rows within their method.** Checkpoint paths are preserved exactly apart from slash normalization. A matching filename does not establish that old and current checkpoint bytes are identical. Historical artifacts lack checkpoint hashes; missing recorded paths are flagged rather than silently substituted. Percentages are metric values multiplied by 100; AUC is unitless. Lower EER/FAR/FRR is better; higher Rank-1/Recall@K is better.

| Method | Model / run | Checkpoint | Evaluation sample | Recorded metrics | Status | Evidence |
|---|---|---|---|---|---|---|
'''
lines = []
for row in rows:
    values = [row[k] for k in ('method','model','checkpoint','sample','result','status')]
    values[2] = '`' + values[2] + '`'
    source = row['source']
    values.append(f'[{source}](../{quote(source, safe="/")})')
    lines.append('| ' + ' | '.join(values) + ' |')
notes = '''

## Evaluation method definitions

- **A — Shared-pair verification:** local cosine comparison of identical test image pairs using `src/training/compare_eer.py`. EER uses ROC interpolation; its test-derived nearest threshold is not the deployed validation threshold. Genuine means the same finger class, not merely the same subject. These are the directly comparable checkpoint rows.
- **B — Historical full evaluation:** `src/training/full_eval.py`; local verification plus real-image gallery versus altered-image probes. Verification sampling and EER implementation differ from A. Identification is closed-set, across all enrolled finger classes. Historical VGG16 epoch 14 is a different run from `vgg16_arcface_v2/best.pt` in A. Some DenseNet and tuning paths are no longer present locally.
- **C — Historical enrollment retrieval:** saved `evaluate_detailed.py` reports, using the full enrollment gallery for each query. These predate the current Pinecone gallery; even equal checkpoint filenames do not make them a current remote measurement. Differences between B and C are retained, not averaged or resolved by choosing the larger score.
- **D — Archive candidate positions:** `src/retrieval/person_evaluation.py` searches the whole corresponding validation/test archive, removes the exact query vector ID, then checks the first, third, and fifth remaining candidates for the correct finger class. Saved `rank2`, `rank4`, `rank6` names refer to this convention. They are individual-position correctness averaged per person, NOT cumulative Recall@2/4/6 and NOT subject-filtered results. Other images of the query finger remain eligible. The two models were sampled independently; seed 42 alone does not prove identical query images.
- **E — Calibration:** validation verification pairs only, with a target FAR of 0.1%. This is not an open-set rejection evaluation or test recognition accuracy.
- **F — Final enrollment gallery:** Pinecone gallery and calibration both use DenseNet121 ArcFace `best.pt`. The full remote evaluation was stopped; its 7,279-query accuracy remains pending. A successful single-query smoke check is not a full accuracy result.
- **G — Current frontend:** searches only the selected subject's classes in the full-image archive and excludes the exact query image and duplicate source paths. This differs from D's all-subject archive search; its scores cannot inherit D's aggregate accuracy. Cosine similarity is not a probability or accuracy percentage.

## Suggested presentation statement

"On a common sample of 1,000 genuine and 1,000 impostor test pairs, DenseNet121 ArcFace best recorded 0% observed EER and ROC-AUC 1.0. Historical full-set identification and archive candidate-position experiments are reported separately because they use different checkpoints or protocols. The final subject-scoped demonstration ranks other images of a known subject and excludes the exact query image."

Regenerate this table with `.venv/Scripts/python.exe src/training/summarize_results.py`. Existing reports are read-only inputs. The JSON companion preserves the same rows for reuse.
'''
(ROOT / 'docs/FINAL_RESULTS.md').write_text(intro + '\n'.join(lines) + notes, encoding='utf-8')
(ROOT / 'docs/final_results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
print(f'Created one comparison table with {len(rows)} rows; shared-pair alignment and person-level aggregates verified.')
print('Methods:', {method: sum(r['method'].startswith(method + ':') for r in rows) for method in 'ABCDEFG'})
