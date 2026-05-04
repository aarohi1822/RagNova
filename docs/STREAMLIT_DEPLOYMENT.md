# Streamlit Deployment Guide

## Quick Fix for Connection Errors

If you see `requests.exceptions.ConnectionError` when running on Streamlit Cloud, follow these steps:

### Problem
Your Streamlit app can't reach the backend API because:
1. Backend is running on `localhost:8000` (not accessible from Streamlit Cloud)
2. `API_URL` environment variable is not set or pointing to wrong address
3. Backend service is not deployed

---

## Solution 1: Local Development (Docker)

Run both services locally using Docker Compose:

```bash
docker-compose up
```

This starts:
- **Backend API** on `http://localhost:8000`
- **Streamlit** on `http://localhost:8501`

---

## Solution 2: Streamlit Cloud Deployment

### Step 1: Deploy Backend to a Public URL

Choose one of these platforms:

#### Option A: Railway.app (Recommended)
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway link  # Select your project
railway up    # Deploy
```

#### Option B: Render
```bash
# Push to GitHub, then:
# 1. Go to render.com
# 2. Create new Web Service
# 3. Connect GitHub repo
# 4. Set Start Command: uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
# 5. Deploy
```

#### Option C: AWS/GCP/Azure
Use their container deployment services (Fargate, Cloud Run, Container Instances)

### Step 2: Set Secrets in Streamlit Cloud

1. Go to **Streamlit Cloud Dashboard**
2. Select your app
3. Click **Settings** (⚙️) in top right
4. Go to **Secrets**
5. Add these two environment variables:

```
OPENAI_API_KEY = sk-your-openai-key-here
API_URL = https://your-deployed-backend.railway.app
```

Replace `https://your-deployed-backend.railway.app` with your actual backend URL.

### Step 3: Redeploy Streamlit App

```bash
git push origin main  # If using GitHub integration
# OR go to Streamlit Cloud and click "Rerun"
```

---

## Solution 3: Local Development (Without Docker)

### Step 1: Install Dependencies
```bash
pip install -e .
```

### Step 2: Set Local Secrets
Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-your-key-here"
API_URL = "http://localhost:8000"
```

### Step 3: Start Backend (Terminal 1)
```bash
python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Start Streamlit (Terminal 2)
```bash
streamlit run frontend/streamlit_app.py
```

---

## Verify Setup

After deployment, check if backend is reachable:

1. Open the Streamlit app
2. Go to sidebar → **Configuration**
3. Click **🔄 Check Backend Connection**
4. Should show: ✅ "Backend is reachable"

If you see ❌, check:
- Backend is deployed and running
- `API_URL` environment variable is correctly set
- No firewall/CORS issues

---

## Troubleshooting

### "ConnectionError: HTTPConnectionPool"
- Backend is not running or not at the URL specified in `API_URL`
- **Fix**: Deploy backend or update `API_URL`

### "Timeout" (slow requests)
- Backend is reachable but slow (cold start, heavy queries)
- **Fix**: Increase timeout or optimize queries

### "CORS errors"
- Backend doesn't allow requests from Streamlit Cloud domain
- **Fix**: Add CORS headers in `backend/app/main.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify Streamlit Cloud domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Secrets not loading
- `.streamlit/secrets.toml` is in `.gitignore` (correct)
- Make sure you've set secrets in Streamlit Cloud dashboard, not just locally
- **Fix**: Commit and push to GitHub, then refresh Streamlit app

---

## Environment Variables Summary

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `API_URL` | Backend API endpoint | `http://localhost:8000` or `https://api.yourapp.com` |

---

## Architecture

```
┌─────────────────────────────────────────────┐
│         Streamlit Cloud / Local             │
│     ┌──────────────────────────────┐       │
│     │   Streamlit Frontend         │       │
│     │  (frontend/streamlit_app.py) │       │
│     └──────────────────────────────┘       │
└─────────────┬──────────────────────────────┘
              │ HTTP POST /chat/ask
              │ (configured via API_URL)
              ▼
    ┌──────────────────────────┐
    │  FastAPI Backend         │
    │  (backend/app/main.py)   │
    │  - /health               │
    │  - /chat/ask             │
    │  - /documents/upload     │
    └──────────────────────────┘
```

