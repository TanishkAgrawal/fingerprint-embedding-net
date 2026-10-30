# fingerprint-embedding-net

Fingerprint embeddings are computed locally with PyTorch and stored/searched in
Pinecone using cosine similarity. Gallery/query CSV files and the manifest remain
local; the manifest records the remote index and namespace.

The [final results comparison](docs/FINAL_RESULTS.md) consolidates saved metrics,
checkpoint paths, evaluation methods, source reports, and pending measurements.

Optional standalone diagnostics and report QA images are preserved in the
[cleanup archive](archive/2026-10-29-cleanup/README.md), with original paths,
SHA-256 checksums, and restoration instructions.

## Pinecone setup

### Final presentation demo

The presentation app now compares images within a selected subject, using the
verified `outputs/embeddings/final_densenet/densenet121_arcface__best-2228dbf0`
archive. Select the test/validation split, subject, finger, and query image.
Pinecone searches only that subject's finger classes; the exact query image and
duplicate source paths are removed. Other images of the same finger remain
eligible. Results include unique image IDs, same-finger indicators and images.
Cosine similarity is not accuracy. This subject-scoped view is distinct from the
all-subject `person_evaluation.py` protocol and does not reproduce its accuracy.
The verification threshold and optional LLM explanation are omitted from this
comparison view because they do not determine within-subject image ranking.

The separate held-out enrollment workflow uses
`outputs/retrieval/densenet121_final` and
`checkpoints/densenet121_arcface/best.pt`. Its build/calibration/evaluation commands are:

```powershell
.venv/Scripts/python.exe -m src.retrieval.build_index --checkpoint checkpoints/densenet121_arcface/best.pt --index-dir outputs/retrieval/densenet121_final --batch-size 64
.venv/Scripts/python.exe -m src.retrieval.calibrate --checkpoint checkpoints/densenet121_arcface/best.pt --out outputs/retrieval/densenet121_final/threshold.json --batch-size 64
.venv/Scripts/python.exe -m src.retrieval.evaluate_detailed --index-dir outputs/retrieval/densenet121_final --batch-size 64
.venv/Scripts/python.exe -m streamlit run demo_app.py
```

The threshold uses validation subjects; the gallery and held-out queries use test
subjects. Historical `densenet121` results use `latest.pt` and remain separate.
Pinecone credentials must be configured as described below.

Install dependencies with `python -m pip install -r requirements.txt`.
The project pins Pinecone SDK 7.x for its vector-index API.
Set these environment variables in the terminal used to run the app (PowerShell):

```powershell
$env:PINECONE_API_KEY = "your-api-key"
$env:PINECONE_INDEX = "fingerprint-embeddings"
$env:PINECONE_CLOUD = "aws"
$env:PINECONE_REGION = "us-east-1"
```

Use your Pinecone project's supported cloud/region. Credentials are read from the
environment or the project root `.env` file (existing environment variables take
precedence). Never commit API keys.
An existing index must use cosine similarity and the checkpoint's embedding
dimension (normally 512). Different embedding dimensions require different indexes.

```powershell
python -m src.retrieval.build_index --checkpoint checkpoints/densenet121/best.pt --index-dir outputs/retrieval/densenet121
python -m src.retrieval.query --index-dir outputs/retrieval/densenet121 --query-number 0 --top-k 5
python -m src.retrieval.evaluate --index-dir outputs/retrieval/densenet121
streamlit run demo_app.py
```

Replace the checkpoint path with the checkpoint you want to use. Building creates
the index if missing and uploads enrollment embeddings plus class IDs/CSV row IDs.
It requires Pinecone access and can incur service charges. Each build uses a new
namespace to prevent stale vectors or other models from affecting retrieval;
previous namespaces remain in Pinecone until you remove them yourself.

Existing FAISS files and historical evaluation reports are retained as artifacts.
Rebuild each gallery before using it with Pinecone; legacy manifests produce a
clear migration error. Training and validation threshold calibration remain local.
Pinecone search is approximate, so rerun evaluation for new retrieval metrics.
Evaluation sends one remote query per image and can take longer than local search.

## Export fingerprint embeddings to files

The saved CNNs already include a trained linear embedding head followed by L2
normalization. ArcFace classification weights are stored separately; inference
uses the embedding head, without classification or retraining.

```powershell
.venv/Scripts/python.exe -m src.retrieval.export_embeddings
```

By default this exports all processed train/val/test images for the selected
DenseNet121 ArcFace best checkpoint under `outputs/embeddings/`. Each model/split directory contains
`embeddings.npy` (float32, one unit vector per row), `metadata.csv` (the same row
order, including image paths and finger identity), and `manifest.json` (checkpoint,
input fingerprint, dimension, progress and completion flag). Outputs are ignored
by Git. Re-running the same command resumes interrupted exports and skips completed
ones; use a new output directory if inputs change. Training images are exported
without augmentation. Keep split boundaries when evaluating retrieval so training
images and query images do not leak into the evaluation gallery.

Use `--all-checkpoints --checkpoint-kind both` to export both best and latest weights, or repeat
`--checkpoint PATH` to select specific files. For a quick check:

```powershell
.venv/Scripts/python.exe -m src.retrieval.export_embeddings --checkpoint checkpoints/resnet18_arcface/best.pt --limit 8 --output-dir outputs/embedding_smoke
```

Load vectors with `numpy.load(".../embeddings.npy", mmap_mode="r")`; row `i`
corresponds to row `i` of `metadata.csv`. This command writes local files and does
not upload them to Pinecone.

## Persist every image embedding in Pinecone (final DenseNet121 model)

The trained DenseNet121 already replaces the ImageNet classifier with a learned
512-dimensional projection and L2 normalization. Use that checkpoint head as-is;
adding a new random head would discard the learned embedding representation.

```powershell
.venv/Scripts/python.exe -m src.retrieval.export_embeddings --checkpoint checkpoints/densenet121_arcface/best.pt --output-dir outputs/embeddings/final_densenet --batch-size 128
.venv/Scripts/python.exe -m src.retrieval.upload_embeddings --export-dir outputs/embeddings/final_densenet/densenet121_arcface__best-REPLACE_WITH_EXPORT_SUFFIX
```

Use the actual directory printed by the export command. All 55,270 currently
processed images are exported: 38,808 train, 8,283 validation, 8,179 test.
Upload creates a separate persistent namespace for each checkpoint/split in
`fingerprint-embeddings`. Each image has its own vector ID, finger identity,
split, source paths, and alteration metadata. Different checkpoints never share
an embedding space. Raw images are not uploaded.

Both steps resume after interruption. Local `pinecone.json` files record the
destination, acknowledged progress, verified remote count and sample read-back.
Re-running verifies completed uploads without uploading those rows again. Keep
the export files and manifests to reproduce the correspondence between vectors
and images. Pinecone stores vectors beyond the lifetime of the Python process.

This full-image archive is separate from the one-image-per-finger evaluation
gallery. Do not search a test image against a gallery containing itself when
reporting recognition accuracy. Use the held-out retrieval workflow above for
model comparisons; archive upload verification is a persistence check.

## Streamlit result explanations with Ollama

In `demo_app.py`, enable **Explain comparison results** in the sidebar and enter
an installed Ollama model name (default: `llama3.2`). Start the local service with
`ollama serve` if it is not already running, and install the model with
`ollama pull llama3.2`. Run the Streamlit app with `streamlit run demo_app.py`,
select a query, and click **Compare subject fingerprints**. A descriptive
explanation appears below the ranked results. If Ollama is unavailable, the
comparison results remain visible with an error message.

The local model receives query metadata and ranked comparison scores, not image
pixels. Its explanation describes a search within the selected known subject;
cosine similarity is not a probability, and no identity or threshold decision
is inferred.

## Contributors

- [StormShadow26](https://github.com/StormShadow26)
- [VaibhavSingh12156](https://github.com/VaibhavSingh12156)
- [TanishkAgrawal](https://github.com/TanishkAgrawal)
