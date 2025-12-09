FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU-only FIRST (10x smaller, 10x faster)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Create directories
RUN mkdir -p outputs uploads && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]



# FROM python:3.10-slim

# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgomp1 \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Create non-root user
# RUN useradd -m -u 1000 appuser && \
#     chown -R appuser:appuser /app

# # Create directories EARLY with proper permissions
# RUN mkdir -p /app/outputs /app/uploads && \
#     chmod -R 755 /app/outputs /app/uploads && \
#     chown -R appuser:appuser /app/outputs /app/uploads

# # Upgrade pip first
# RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# # Copy requirements
# COPY requirements.txt .

# # Install PyTorch CPU-only FIRST (10x smaller, 10x faster)
# RUN pip install --no-cache-dir \
#     torch==2.1.0+cpu \
#     --index-url https://download.pytorch.org/whl/cpu

# # Install remaining packages
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY --chown=appuser:appuser . .

# # Ensure permissions AGAIN after copy (in case source has these folders)
# RUN mkdir -p outputs uploads && \
#     chmod -R 755 outputs uploads && \
#     chown -R appuser:appuser outputs uploads

# USER appuser

# EXPOSE 8000

# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]