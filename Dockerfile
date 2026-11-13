FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY .streamlit/config.toml .streamlit/config.toml
COPY frontend ./frontend

EXPOSE 8501

# Pass the Groq key at runtime: docker run -e GROQ_API_KEY=... -p 8501:8501 <image>
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
