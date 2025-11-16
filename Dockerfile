FROM python:3.10.19-slim

WORKDIR /app

# Install build tools and scientific libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Flask port
EXPOSE 9698

# Start the app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:9698", "predict:app"]

