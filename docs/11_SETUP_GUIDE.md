# Setup and Installation Guide

> **Maintainer:** Saumya Mishra (MID-07). **Goal:** a new teammate can get from zero to "tests pass and data checked" without asking anyone.
> **Verification:** QA branch commit `00de4267632e0302f587c46187e03a5f56b52874` passed the Python 3.11 editable install and full pytest workflow (**144 passed in 14.59s**) and the Docker image build in [CI run #43574590439](https://github.com/Saumya-Mishra-536/Higgs_Boson_Capstone/actions/runs/43574590439). The separate clean macOS Apple Silicon clone (Python 3.15, **143 passed**) and CI run #26 are earlier baseline evidence. Independent model retraining, full manual notebook execution and module-owner integration review remain unverified. See `docs/12_QA_TEST_PLAN.md`.

---

## 1. What you need

| Requirement | Why | Check |
|---|---|---|
| Git | Clone and make pull requests | `git --version` |
| Python 3.10 or newer (3.12 and 3.14 verified) | Everything is Python | `python --version` |
| About 4 GB free disk for the compressed dataset, more if you decompress or cache subsets | The HIGGS file is about 2.6 GB compressed | `df -h` |
| 8 GB RAM minimum recommended for 250k-row experiments (measured peak memory < 1.5 GB in streaming mode) | Subset-first workflow | n/a |
| A GitHub account that Krishna has invited | Repository access | n/a |
| Optional: an NVIDIA GPU | Faster MLP training; CPU is acceptable for the subset | `nvidia-smi` |

---

## 2. Get the code **[works now]**

```bash
git clone https://github.com/Saumya-Mishra-536/Higgs_Boson_Capstone.git
cd Higgs_Boson_Capstone
git switch main
git pull --ff-only
git switch -c docs/<yourname>-<topic>
```

The canonical repository for this course project is `Saumya-Mishra-536/Higgs_Boson_Capstone`. Never commit directly to `main`.

---

## 3. Create an isolated environment **[works now]**

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

---

## 4. Install dependencies **[verified on a clean clone]**

Use the project metadata as the canonical install path. It installs the runtime, dashboard, training and development/test dependencies declared by `pyproject.toml`:

```bash
python -m pip install -e ".[dev]"
```

The clean-clone check on macOS Apple Silicon with Python 3.15 completed this command successfully. GitHub Actions also completed it on Python 3.11. Package downloads experienced transient network timeouts but resumed successfully.

`requirements.txt` remains a proposed lower-bound list and is not the canonical tested install path. PyTorch is a large dependency; dashboard-only use can install `.[dashboard]`, while the full test suite should use `.[dev]`.

Verify the environment:

```bash
python -c "import numpy, pandas, sklearn, xgboost, torch; print('imports ok')"
```

---

## 5. Download and check the data **[works now]**

You can download and prepare the dataset automatically or manually:

**Option A: Automated script (recommended)**
```bash
python scripts/download_data.py --prepare-subset
```
This fetches the archive into `data/raw/HIGGS.csv.gz`, checks the SHA-256 hash, and extracts the 250k benchmark subset into `data/processed/higgs_sample_250k.csv.gz`.

**Option B: Manual download**
1. Open the **official** dataset page: <https://archive.ics.uci.edu/dataset/280/higgs>.
2. Download the dataset file (`HIGGS.csv.gz`, about 2.6 GB). If the page offers a zip wrapper, extract the `.csv.gz` from it.
3. Move it to `data/raw/HIGGS.csv.gz`. Do **not** rename the file and do **not** commit it (`.gitignore` blocks it).

4. Record a checksum for the data card:

   ```bash
   shasum -a 256 data/raw/HIGGS.csv.gz        # macOS / Linux
   Get-FileHash data\raw\HIGGS.csv.gz         # Windows
   ```

5. Run the smoke test (streams only the first rows):

   ```bash
   python tests/integration/smoke_check_raw_file.py data/raw/HIGGS.csv.gz --rows 100000
   ```

   Expected: `RESULT: PASS`. A failure lists exactly what is wrong (wrong column count, bad labels, missing values, header row).

6. Copy the printed label counts and b-tag values into your PR or the data card. Report only the numbers the script printed.

**Do not** use `ucimlrepo.fetch_ucirepo(id=280)` or a pandas `read_csv` of the whole file on a laptop; the dataset has about 11 million rows. Chunked loading is Aditya's task (MID-01). Using a third-party mirror is not allowed without Krishna's approval (ADR-0001).

---

## 6. Run the automated checks **[works now]**

```bash
python -m pytest tests/unit -q
```

What this covers today:

| Test file | Checks |
|---|---|
| `test_repo_hygiene.py` | Required files exist, no data/model/secret files, no oversized files, `.gitignore` protections, working Markdown links, backlog format, no unsourced metric numbers in docs |
| `test_notebook_style.py` | Every code cell is preceded by a Markdown explanation, no tutorial comments, short code cells, no large outputs |
| `test_feature_dictionary.py` | The 1 + 21 + 7 column layout matches the UCI order |

Expect all tests to pass on a clean checkout. If one fails, read the assertion message; it names the file and line.

---

## 7. Run the pipeline **[works now]**

The baseline models and data loaders are implemented. Reproduction should be recorded against the exact commit, data manifest and environment used.

| Step | Command | Status & Owner | Expected output |
|---|---|---|---|
| Prepare a 250k-row subset | `PYTHONPATH=src python -m higgs_event_lab.data --source data/raw/HIGGS.csv.gz --output data/processed/higgs_sample_250k.csv.gz --rows 250000` | **[works now]** Aditya (MID-01) | `data/processed/higgs_sample_250k.csv.gz` and `.manifest.json` |
| Train classical baselines | `PYTHONPATH=src python -m higgs_event_lab.classical --data data/processed/higgs_sample_250k.csv.gz --rows 250000 --all-groups --output-dir outputs` | **[works now]** Aditya (MID-03) | Measured `metrics.json` and `test_predictions.csv` for each feature group in `outputs/logreg_<group>/` |
| Train the MLP | `PYTHONPATH=src python -m higgs_event_lab.neural --data data/processed/higgs_sample_250k.csv.gz --rows 250000 --all-groups --output-dir outputs` | **[works now]** Rachit (MID-04) | Measured `metrics.json` and `test_predictions.csv` for each feature group in `outputs/mlp_<group>/` |
| Shared split and artifact contract | `PYTHONPATH=src python -m pytest tests/unit/test_protocol.py tests/unit/test_neural.py -q` | **[works now]** Krishna (MID-05) | Contract and MLP unit-test result |

## 8. Run the dashboard

From the repository root, use Python 3.10 or newer:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
streamlit run streamlit_app.py
```

The `dev` extra includes Streamlit, plotting, training and test dependencies.
For dashboard-only use, install `.[dashboard]`. Neural inference requires the
optional `training` extra. Launch from the repository root so artifact paths resolve.

The root `streamlit_app.py` is the canonical launcher. `app/main.py` implements
the application; the legacy launcher delegates to it. The dashboard reads saved
`outputs/<run_id>/metrics.json` and `test_predictions.csv` packages using the
accepted `higgs-experiment-v1` schema. Startup never generates benchmark results.
Missing artifacts show an empty state; invalid runs show actionable diagnostics.

For Streamlit Cloud, choose `streamlit_app.py` and provide real saved run packages.
Do not generate sample artifacts as experiment evidence.

---

## 9. Notebooks **[works now for the physics notebook]**

```bash
jupyter lab notebooks/00_physics/00_higgs_physics_and_dataset_intro.ipynb
```

Follow the notebook policy in `notebooks/README.md`: explanations go in Markdown cells, code cells stay short and comment-free, and reusable logic moves to `src/higgs_event_lab/`.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pytest` | Environment not activated or development dependencies not installed | Activate `.venv`, run `python -m pip install -e ".[dev]"` |
| `test_no_data_model_or_system_junk_files_are_present` fails | A `.DS_Store`, `.gz`, `.pt` or similar file sits inside the repository folder | Delete it or move it under an ignored path; confirm `.gitignore` is present |
| Smoke test: `File not found` | Wrong path or file name | Check `ls data/raw/` |
| Smoke test: `rows do not have 29 columns` | Truncated download or wrong file | Re-download; compare the checksum |
| Smoke test: `First row looks like a header` | A different edition or a converted file | Stop and tell Krishna; do not silently adapt |
| Out-of-memory while loading | Whole-file read | Use chunked reads (`chunksize`) and a subset |
| Git refuses a push of a large file | Data accidentally staged | `git rm --cached <file>`, confirm `.gitignore`, recommit |
| PowerShell cannot activate the venv | Execution policy | See section 3 |

## 11. Before opening a pull request

- [ ] `python -m pytest tests/unit -q` passes (paste the real output in the PR).
- [ ] No data, model weights, `.env` or notebook outputs with large data are staged (`git status`).
- [ ] Docs and `team/CHANGELOG.md` updated for any user-visible change.
- [ ] You linked the issue and named a reviewer who is not you.
