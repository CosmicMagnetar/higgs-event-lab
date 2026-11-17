# `00_START_HERE.md`

## Start Here — Team Leader Setup

### Purpose of This Repository

This repository is the shared home for a five-person course project. It establishes a clear scientific scope, folder structure, ownership boundaries, and review process before model implementation begins.

---

### Team Leader Checklist for Krishna

#### One-Time Repository Setup

* **Create Repository:** Create a new private repository named `higgs-event-lab` under the intended GitHub account (keep separate from the legacy repository).
* **Initial Commit:** Upload the files from this starter scaffold as the initial commit.
* **Collaborator Access:** Add Saumya Mishra, Aditya Rana, Rachit Gupta, and Ronit Singh as collaborators using their exact GitHub usernames (collect usernames rather than guessing).
* **Branch Policy:** Maintain `main` as the stable branch. Require all work to occur on issue branches and merge via pull requests.
* **Labels:** Create labels according to `docs/04_GITHUB_WORKFLOW.md`:
* **Areas:** `area:docs`, `area:data`, `area:classical-ml`, `area:deep-learning`, `area:evaluation`, `area:app`, `area:deployment`
* **Types:** `type:bug`, `type:feature`, `type:research`
* **Priorities:** `priority:p0`, `priority:p1`, `priority:p2`
* **Stages:** `stage:foundation`, `stage:midsem`, `stage:endsem`


* **Project Board:** Set up a GitHub Project board with six status columns:
`Backlog` | `Ready` | `In Progress` | `In Review` | `Blocked` | `Done`
* **Task Backlog:** Assign owners using `team/TASK_BACKLOG.csv` and populate deadlines once confirmed with course dates.
* **Schema Agreement:** Agree on a single artifact/output schema prior to separate model and app module development.

---

### What Not to Do Yet

* **No Data Tracking:** Do not upload the HIGGS dataset or large prediction files.
* **No Fragmented Workflows:** Do not start five incompatible notebooks or separate apps.
* **No Fabricated Metrics:** Do not add fake model scores to screenshots or README badges.
* **No Premature Scale:** Do not attempt full-dataset training before the starter experiment is reliable.
* **No Early Deployment:** Do not deploy until the test suite, data-source display, and limitation statements are complete.

---

### First Team Meeting Agenda (30–40 Minutes)

1. **Goal & Scope (5 min):** Agree on project goal and scientific boundaries.
2. **Folder Contracts (10 min):** Review folder purposes and confirm module ownership.
3. **Workflow Rules (5 min):** Agree on repo name, visibility, branch/PR rules, and project board.
4. **Deliverable Clarification (5 min):** Confirm mid-semester requirements (proposal, demo, notebook, or all three).
5. **Issue Creation (10 min):** Draft the first 5–8 issues and define clear "Definition of Done" criteria.
6. **Wrap-Up (5 min):** Record key decisions, blockers, and set the next check-in time.

---

### Definition of Done for Setup Phase

* [ ] A new repository exists at `<OWNER>/higgs-event-lab` with the initial scaffold committed.
* [ ] All five team members have accepted collaborator invites.
* [ ] Folder contracts and team ownership are clear to all members.
* [ ] No raw datasets or secrets are tracked in git history.
* [ ] Starter issue backlog and initial meeting notes are created.
* [ ] Team members can successfully clone the repo and open a test PR.

---

### Suggested Local Setup Commands

Once the empty GitHub repository is created, extract the starter files, open your terminal in the root directory, and execute:

```bash
git init
git add .
git commit -m "chore: initialize Higgs Event Lab project scaffold"
git branch -M main
git remote add origin https://github.com/<OWNER>/higgs-event-lab.git
git push -u origin main

```

> **Note:** Replace `<OWNER>` with the target GitHub username. If the GitHub repository was initialized with a default `README` or license, reconcile git histories prior to pushing.
