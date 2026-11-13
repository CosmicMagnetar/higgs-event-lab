# Deployment

**Owner:** Krishna (Team Lead).

Deployment starts only after:
- The real artifact schema (`higgs-experiment-v1`) is stable.
- The Streamlit app loads artifacts without training on startup.
- Smoke tests pass.
- No secrets are tracked in Git.
- The scientific limitation statement is visible in the dashboard.

---

## 1. Provider Selection

The provider/platform will be selected later based on the team's course requirements, account limitations, resource needs, and availability. Candidates include Streamlit Community Cloud, Docker on a VPS, or a managed container platform.

---

## 2. Local Deployment

### Docker

Build and run locally:

```bash
# Build the image
docker build -t higgs-event-lab -f deployment/Dockerfile .

# Run the container
docker run -p 8501:8501 higgs-event-lab
```

The dashboard will be available at `http://localhost:8501`.

### Local Shell Script

```bash
# Deploy locally with the provided script
./deployment/deploy_local.sh
```

---

## 3. Streamlit Community Cloud (Recommended for Demo)

One-click deployment:
1. Connect repository `CosmicMagnetar/higgs-event-lab` on branch `main`.
2. Main file path: `streamlit_app.py`.
Provide genuine saved run packages under `outputs/`. Startup never generates results; missing artifacts show an empty state.

---

## 4. Environment Configuration

- Do **not** put real secrets in this file or in the repository.
- Use platform environment settings and local `.env` files (excluded by Git) for any required credentials.
- Set `HIGGS_REPO_ROOT` if running notebooks from a cloned location other than the current directory.

---

## 5. Rollback

If a deployment fails:
1. Revert to the previous stable commit on `main`.
2. Re-run smoke tests against the artifact schema.
3. Confirm the dashboard loads existing `outputs/` artifacts without retraining.

See `deployment/DEPLOYMENT_RUNBOOK.md` for the full runbook and `deployment/Dockerfile` for the container definition.

---
