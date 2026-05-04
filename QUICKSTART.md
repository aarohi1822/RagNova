# Quick Start Guide

## 🚀 Get Started in 2 Minutes

### Option 1: Docker (Recommended)

```bash
# 1. Copy environment file
cp .env.example .env

# 2. Edit .env with your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here

# 3. Start both backend and frontend
docker-compose up
```

Then open:
- **Frontend:** http://localhost:8501
- **Backend API:** http://localhost:8000/docs (Swagger UI)

---

### Option 2: Local Python

```bash
# Terminal 1: Backend
python -m uvicorn app.main:app --app-dir backend --reload

# Terminal 2: Streamlit
streamlit run frontend/streamlit_app.py
```

Create `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-your-key-here"
API_URL = "http://localhost:8000"
```

---

### Option 3: Streamlit Cloud

1. **Deploy backend first:**
   - Use Railway, Render, or AWS
   - Get your backend URL (e.g., `https://api.myapp.com`)

2. **Deploy Streamlit:**
   - Push code to GitHub
   - Go to streamlit.io/cloud
   - Connect your repo

3. **Set secrets in Streamlit Cloud:**
   - Settings → Secrets
   - Add `OPENAI_API_KEY` and `API_URL`

---

## 🔗 Verify Connection

After starting the app, go to sidebar **Configuration** and click **🔄 Check Backend Connection**.

Should show ✅ if everything is working.

---

## 📚 More Info

- [Full Deployment Guide](docs/STREAMLIT_DEPLOYMENT.md)
- [Backend API Docs](http://localhost:8000/docs) (when running locally)
- [Architecture](docs/ARCHITECTURE.md)
