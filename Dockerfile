# ✅ Force AMD64 architecture (important for evaluators)
FROM --platform=linux/amd64 python:3.10-slim

# ✅ Set the working directory inside the container
WORKDIR /app

# ✅ Install system-level dependencies including build tools for compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ✅ Upgrade pip first
RUN pip install --upgrade pip

# ✅ Copy only requirements first for caching
COPY requirements.txt .

# ✅ Install Python dependencies with more verbose output
RUN pip install --no-cache-dir --verbose -r requirements.txt

# ✅ Copy the rest of your project files
COPY . /app

# ✅ Run setup first, then main script when container starts
CMD ["bash", "-c", "python setup.py && python challenge1b_main.py"]