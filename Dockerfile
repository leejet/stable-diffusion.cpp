# Runtime only (Ubuntu 24.04 + CUDA 12.6 Runtime)
# Compatible with Host CUDA 12.9 drivers
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install System Deps (Python + Libraries for sd.cpp binary)
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    libgomp1 libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Virtual Environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 3. Install Python Deps via requirements.txt
# We copy ONLY the requirements file first to leverage Docker caching
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 4. Copy the sd-cli binary (assuming it's built separately)
COPY build/bin/sd /usr/local/bin/sd

# 5. Set Workspace
WORKDIR /sdcpp/server

# 6. Run Server (Reload enabled for dev)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]# Runtime only (Ubuntu 24.04 + CUDA 12.6 Runtime)
# Compatible with Host CUDA 12.9 drivers
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install System Deps (Python + Libraries for sd.cpp binary)
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    libgomp1 libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Virtual Environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 3. Install Python Deps via requirements.txt
# We copy ONLY the requirements file first to leverage Docker caching
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 4. Copy the sd-cli binary (assuming it's built separately)
COPY build/bin/sd /usr/local/bin/sd

# 5. Set Workspace
WORKDIR /sdcpp/server

# 6. Run Server (Reload enabled for dev)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
