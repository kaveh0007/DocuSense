# Lightweight image for CPU inference
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./

# Install build deps for some Python wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Streamlit default port
EXPOSE 8501

# By default, run the UI
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
