FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
COPY hf_space/requirements.txt hf_requirements.txt
RUN pip install --no-cache-dir -r hf_requirements.txt

# Copy application code
COPY hf_space/ ./hf_space/
COPY src/ ./src/
COPY config.yaml .

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "hf_space/app.py"]
