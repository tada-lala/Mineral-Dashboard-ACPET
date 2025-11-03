# Use a slim Python base image
FROM python:3.10-slim

WORKDIR /app

# Install only system packages once
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first (to leverage caching)
COPY requirements.txt .

# Install Python dependencies (this layer will be cached if requirements.txt doesn’t change)
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Now copy your app code and data
COPY app.py .
COPY "final1.csv" .

EXPOSE 8050

CMD ["gunicorn", "app:server", "-b", "0.0.0.0:8050"]

