02_REPOSITORY_STRUCTURE
Repository Structure and Folder Contracts
This tree is the agreed target layout. Empty folder contracts are represented by README files because Git does not track empty directories. The first commit is a scaffold; source code and notebooks will be added through later issues/PRs.
higgs-event-lab/
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── .editorconfig
├── requirements.txt
├── docs/
│   ├── 00_START_HERE.md
│   ├── 01_PROJECT_CHARTER.md
│   ├── 02_REPOSITORY_STRUCTURE.md
│   ├── 03_SCIENTIFIC_SCOPE_AND_DATA.md
│   ├── 04_GITHUB_WORKFLOW.md
│   ├── 05_MILESTONES_AND_ACCEPTANCE.md
│   ├── 06_AI_PROMPTS.md
│   ├── 07_REFERENCES.md
│   ├── 08_PHYSICS_BACKGROUND.md
│   ├── 09_DATASET_GUIDE.md
│   ├── 10_OBJECTIVES_AND_METHODOLOGY.md
│   ├── 11_SETUP_GUIDE.md
│   ├── 12_QA_TEST_PLAN.md
│   ├── feature_dictionary.csv
│   └── decisions/
│       └── ADR-0001-data-source-and-project-boundary.md
├── team/
│   ├── TEAM_CHARTER.md
│   ├── TASK_BACKLOG.csv
│   ├── MEETING_NOTES_TEMPLATE.md
│   ├── BUG_TRACKER.md
│   └── CHANGELOG.md
├── data/
│   ├── raw/README.md
│   ├── interim/README.md
│   ├── processed/README.md
│   └── external/README.md
├── notebooks/
│   ├── README.md
│   ├── 00_physics/      (README.md, 00_higgs_physics_and_dataset_intro.ipynb)
│   └── 01_midsem/
├── src/higgs_event_lab/README.md
├── tests/
│   ├── README.md
│   ├── unit/            (README.md, test_*.py)
│   └── integration/     (README.md, smoke_check_raw_file.py)
├── app/README.md
├── reports/             (README.md, HiggsLens_Final_Report.md)
├── outputs/README.md
├── artifacts/README.md
├── deployment/README.md
└── .github/
    ├── PULL_REQUEST_TEMPLATE.md
    └── ISSUE_TEMPLATE/
        ├── bug_report.md
        └── feature_request.md
​
What belongs where
Path
Purpose
Primary owner
docs/
Scientific context, decision records, protocols, user docs
Saumya maintains docs; Krishna approves scientific/product decisions
team/
Roles, backlog, meeting notes and change log
Saumya maintains; Krishna sets priorities
data/raw/
Original downloaded data, never committed
Aditya
data/interim/
Temporary cleaned/intermediate local data, never committed
Aditya
data/processed/
Reproducibly prepared local data, normally not committed
Aditya
data/external/
Other explicitly documented sources, never silently mixed
Aditya + Krishna approval
notebooks/00_physics/
Guided explanation/EDA notebooks, if needed
Saumya for teaching text; domain owner for code
notebooks/01_midsem/
Small, deadline-oriented prototype workflow
Krishna coordinates, Aditya/Rachit own model cells
src/higgs_event_lab/
Reusable importable code: data, features, models, evaluation, artifacts
Aditya data/classical; Rachit MLP; Krishna shared evaluation contract
tests/
Unit and integration tests
Each code author writes tests; Saumya coordinates/triages QA
app/
User-facing Streamlit UI using saved outputs
Ronit
reports/
Written report source, figures selected for paper and references
Krishna synthesizes; Saumya edits/maintains docs; all supply findings
outputs/
Local model outputs and experiment metrics, Git-ignored by default
Each experiment owner; Krishna locks final release artifacts
artifacts/
Local trained model/checkpoint artifacts; Git-ignored
Rachit models; Krishna signs off release package
deployment/
deployment config, health/readiness checklist and runbook
Krishna
.github/
PR/issue templates; CI can be added after tests and package exist
Krishna + Saumya
Folder design rules
No model-training implementation in the first scaffold.
No notebooks containing duplicated copies of the same reusable logic once that logic is moved into src/.
No large raw data, model weights or run outputs committed by default.
The dashboard reads saved, versioned experiment outputs; it should not train the whole benchmark on app startup.
One agreed artifact schema must be documented before Rachit, Aditya and Ronit integrate independent modules.
