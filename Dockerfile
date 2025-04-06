FROM python:3.11-slim

WORKDIR /app

# ✅ 1. Install OpenCV dependency
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

# ✅ 2. Copy necessary files
COPY requirements.txt .
COPY endpoint.py .
COPY skin_cancer_detection_model.h5 .

# ✅ 3. Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# ✅ 4. Expose port for Azure
EXPOSE 8000

# ✅ 5. Run with production-ready server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "endpoint:app"]
