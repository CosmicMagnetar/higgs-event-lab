# Deployment Runbook

**Owner:** Krishna (Team Lead). **Scope:** Environment configuration, containerized deployment, smoke testing, and rollback procedure (`END-06`).

## 1. Prerequisites and Safety Rules

- Never commit raw datasets (`HIGGS.csv.gz`), model binaries (`*.pt`), or secrets to the repository.
- The dashboard container must start without triggering background model training.
- All visualizations load pre-generated experiment artifacts from `outputs/`.
- If no artifacts exist, the dashboard displays a clear empty state with reproduction instructions.

## 2. Local Containerized Deployment

### Build the Image
```bash
docker build -t higgs-event-lab:latest -f deployment/Dockerfile .
```

### Run the Container
```bash
docker run -d --name higgs-app -p 8501:8501 higgs-event-lab:latest
```

### Verify Container Health
```bash
curl -f http://localhost:8501/_stcore/health
```

Access the UI at `http://localhost:8501`.

### Stop and Remove
```bash
docker stop higgs-app && docker rm higgs-app
```

## 3. Cloud Deployment (Streamlit Community Cloud)

Streamlit Community Cloud hosts the interactive dashboard directly from GitHub with zero container build overhead:

1. Push all release commits to `https://github.com/CosmicMagnetar/higgs-event-lab` on branch `main`.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
3. Select:
   - **Repository:** `CosmicMagnetar/higgs-event-lab`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Click **Deploy!**
Provide genuine saved run packages under `outputs/`. Startup never generates results; missing artifacts show an empty state.

## 4. Smoke Test Checklist

- [ ] Container or web service starts up in less than 30 seconds.
- [ ] Health check endpoint (`/_stcore/health`) returns status 200.
- [ ] Title, scientific scope, and limitation notices render on the main page.
- [ ] Sidebar allows selecting model families and feature groups.
- [ ] Plots (ROC curves, PR curves, calibration) render interactively.
- [ ] Empty state renders gracefully when no artifacts are found.

## 5. Rollback Procedure

If a deployed version experiences unexpected regressions or runtime errors:

1. Identify the last verified stable release commit on `main`:
   ```bash
   git log --oneline -n 5
   ```
2. Revert to the stable commit or previous release tag:
   ```bash
   git checkout <stable-commit-hash>
   ```
3. Re-build and re-deploy the container:
   ```bash
   docker build -t higgs-event-lab:rollback -f deployment/Dockerfile .
   docker stop higgs-app && docker rm higgs-app
   docker run -d --name higgs-app -p 8501:8501 higgs-event-lab:rollback
   ```
4. Record the incident, root cause, and remediation steps in `team/BUG_TRACKER.md`.
