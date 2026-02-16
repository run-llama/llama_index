FROM node:22-slim

# System dependencies for OpenCV and MediaPipe (headless — no X11)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-venv \
    make \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g pnpm

# Python dependencies — headless OpenCV (no GUI), MediaPipe, NumPy
# Using --break-system-packages for container-only installs
RUN pip3 install --break-system-packages \
    opencv-python-headless \
    mediapipe \
    numpy \
    Pillow

# Install the Face Lock biometric package
# In production, pip install from PyPI. In dev, copy and install local.
COPY ../llama_index /opt/face_lock_src/llama_index
COPY ../pyproject.toml /opt/face_lock_src/pyproject.toml
COPY ../README.md /opt/face_lock_src/README.md
RUN pip3 install --break-system-packages /opt/face_lock_src

WORKDIR /app

# Node.js dependencies
COPY package.json pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile || pnpm install

# Copy application code
COPY . .

# Create workspace directories
RUN mkdir -p /app/workspace/temp /app/config

EXPOSE 3000

CMD ["pnpm", "start"]
