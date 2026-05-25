# Vercel + Render Deployment Guide

This repository now uses a split deployment:

- Vercel hosts the lightweight frontend
- Render hosts the Python backend API
- models and runtime inference stay on Render

## Architecture

- `index.html`: static frontend served by Vercel
- `api/health.js`: tiny Vercel proxy for backend health
- `api/generate.js`: tiny Vercel proxy for quiz generation
- `backend/app.py`: Flask backend for Render
- `backend/requirements.txt`: backend-only runtime dependencies
- `render.yaml`: Render service definition
- `vercel.json`: serves the static frontend
- `.vercelignore`: excludes local-only folders and large non-runtime assets from deployment

## Before You Deploy

Make sure these files are committed and pushed to your GitHub repository:

- `index.html`
- `api/health.js`
- `api/generate.js`
- `vercel.json`
- `.vercelignore`
- `backend/app.py`
- `backend/requirements.txt`
- `render.yaml`
- `src/inference.py`
- `src/runtime_helpers.py`
- `requirements.txt`

## GitHub Repo

Deploy this repository:

`muhammadyahya321/AI-ML-Project`

## Step 1: Deploy The Backend On Render

1. Sign in at `https://render.com` with GitHub.
2. Click `New` -> `Blueprint`.
3. Select `muhammadyahya321/AI-ML-Project`.
4. Render should detect `render.yaml`.
5. Create the service.
6. Wait for the backend build to finish.
7. Open the backend URL and test:

`https://your-render-service.onrender.com/api/health`

You should get JSON like:

```json
{"ok": true, "demo_mode": false}
```

## Step 2: Configure Vercel To Point At Render

By default, the Vercel proxy falls back to the Render service name from this repo:

`https://ai-ml-project-backend.onrender.com`

If your Render service uses that exact name, you do not need to set any Vercel environment variable.

Only add a Vercel environment variable if your Render URL is different. Then in Vercel project settings:

1. Go to `Settings` -> `Environment Variables`
2. Add either:
   - Name: `BACKEND_URL`
   - Value: your Render base URL
3. Or add:
   - Name: `RENDER_SERVICE_NAME`
   - Value: your Render service name without `https://` or `.onrender.com`
4. Save the variable

## Step 3: Deploy The Frontend On Vercel

1. Sign in at `https://vercel.com` using your GitHub account.
2. Click `Add New` -> `Project`.
3. Import `muhammadyahya321/AI-ML-Project`.
4. If Vercel asks for framework, choose `Other`.
5. Leave the Root Directory as the repository root.
6. Do not set a custom build command.
7. Do not set an output directory.
8. Click `Deploy`.

## Recommended Project Settings

After import, verify these:

- Framework Preset: `Other`
- Install Command: default
- Build Command: empty
- Output Directory: empty

## After Deployment

When both deployments finish:

1. Open the generated Vercel URL.
2. Paste a passage into the text area.
3. Click `Generate Quiz`.
4. Confirm that the quiz, options, and hints appear.

## If GitHub Changes Do Not Show Up

1. Commit your local changes.
2. Push them to GitHub.
3. In Vercel, open the project.
4. Click `Redeploy` on the latest deployment if needed.

## Notes About Data And Models

- Your committed model artifacts in `models/` are small enough for this deployment style.
- `data/raw/*.csv` is ignored by Git, so the deployed app should not depend on those CSV files.
- If some trained artifacts are missing on Render, the backend still runs in demo mode.

## Local Smoke Tests

Test the backend locally with:

```bash
python -m flask --app backend/app.py run --port 8000
```

Then test the frontend by opening `index.html` through a simple local server or after deploying to Vercel.

If you want to test the frontend against a local backend, point requests to a local server or deploy the backend first.

## Why This Works Better

- Vercel no longer bundles Python ML dependencies
- Render handles the heavier inference runtime
- your dataset/model approach can stay intact
- the frontend remains fast and easy to redeploy

```text
http://127.0.0.1:5000
```
