FROM python:3.11-slim

# Install system dependencies required by OpenCV and Tesseract
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn python-multipart sqlalchemy pytesseract

# Copy the rest of the application
COPY . .

# Ensure uploads directory exists
RUN mkdir -p uploads

# Expose port for Cloud Run
EXPOSE 8080

# Command to run the application (Cloud Run sets PORT env variable to 8080 by default)
CMD ["uvicorn", "main_app:app", "--host", "0.0.0.0", "--port", "8080"]
