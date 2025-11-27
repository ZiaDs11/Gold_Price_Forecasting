# Use a lightweight base image for the final stage
FROM python:3.10-slim as base

# Set environment variables for the application paths (Absolute paths in container)
ENV MODEL_PATH=/app/models/gold_lstm_model.h5
ENV SCALER_PATH=/app/models/scaler.save
ENV CSV_PATH=/app/data/goldstock.csv

# Install production dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY app/ app/
COPY models/ models/
COPY data/ data/

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]