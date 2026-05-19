# Vercel Deployment Guide

This repository now includes a Vercel-compatible Flask entrypoint in `api/index.py` and routing in `vercel.json`.

## Important Note

The original UI in `ui/app.py` is a Streamlit app. Vercel does not host Streamlit apps directly the way `streamlit run` expects, so the Vercel deployment uses a lightweight Flask frontend that reuses the existing inference pipeline from `src/inference.py`.

## Files Added For Vercel

- `api/index.py`: Vercel Python entrypoint
- `vercel.json`: rewrites the site root and API requests to the Flask app

## Before You Deploy

Make sure these files are committed and pushed to your GitHub repository:

- `api/index.py`
- `vercel.json`
- `src/inference.py` update
- `requirements.txt`

## GitHub Repo

Deploy this repository:

`muhammadyahya321/AI-ML-Project`

## Steps On Vercel

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

## Python Version

If Vercel asks for a Python version, use Python `3.11`.

## After Deployment

When deployment finishes:

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
- If some trained artifacts are missing during deployment, the app will still run in demo mode.

## Local Smoke Test

You can test the Vercel entrypoint locally with:

```bash
python -m flask --app api/index.py run
```

Then open:

```text
http://127.0.0.1:5000
```
