FROM python:3.10

# Set working directory
WORKDIR /code

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into container
COPY . /code

# Expose HF Spaces Docker port
EXPOSE 7860

# Start FastAPI backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
