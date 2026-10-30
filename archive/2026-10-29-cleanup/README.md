# Archived diagnostics and report inspection files

Archived on 29 October 2026. Eight files were moved without changing their contents.
`manifest.json` records each original path, archive path, reason, byte size, and
SHA-256 hash. Paths in the manifest are relative to the repository root.

- `diagnostics/`: standalone border statistics, histogram plotting, and processed-array integrity scripts. They are optional utilities, not application modules or automated tests.
- `report-inspection/`: five report QA page images and contact sheets. The final PDF, report builder, bibliography, and editable diagrams remain in `output/pdf/fingerprint_report/`.

Checkpoints, datasets, all experiment logs, historical FAISS indexes, embedding
archives, metrics, evaluation reports, final comparison documents, and tests were
left in their existing locations. Python caches and the virtual environment are
ignored generated files and were not included in the archive.

## Run an archived diagnostic

Run from the repository root, with the project on Python's import path. For example:

```powershell
$env:PYTHONPATH = (Get-Location).Path
.venv/Scripts/python.exe archive/2026-10-29-cleanup/diagnostics/demo_test.py
.venv/Scripts/python.exe archive/2026-10-29-cleanup/diagnostics/eval_histogram.py --help
.venv/Scripts/python.exe archive/2026-10-29-cleanup/diagnostics/src/data/checkup.py
```

These retain their original behavior: some read the full dataset or display plots.

## Restore an individual file

Use the manifest to identify the original and archived paths. From the repository
root, confirm the original destination is absent, then copy the archived file back:

```powershell
if (Test-Path -LiteralPath 'demo_test.py') { throw 'Original destination already exists' }
Copy-Item -LiteralPath 'archive/2026-10-29-cleanup/diagnostics/demo_test.py' -Destination 'demo_test.py'
```

The archive copy remains available. Verify its contents using `Get-FileHash` and
the corresponding SHA-256 entry in `manifest.json`.
