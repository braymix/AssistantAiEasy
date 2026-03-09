# Installation Guide

## Prerequisites

### All Profiles

- **Docker** 24+ and **Docker Compose** v2
- **Git** for cloning the repository
- **4 GB RAM** minimum (8+ GB recommended)

### Mini Profile (single machine)

- Any x86_64 or ARM64 machine (mini PC, laptop, server)
- No GPU required — Ollama runs on CPU
- 10 GB free disk space

### Enterprise Profile

- NVIDIA GPU with CUDA 12+ (for vLLM)
- **nvidia-container-toolkit** installed
- PostgreSQL-compatible storage (SSD recommended)
- 32+ GB RAM recommended
- 50+ GB free disk space

---

## Mini Profile Setup

### 1. Clone and configure

```bash
git clone <repository-url>
cd knowledgehub

# Copy and edit configuration
cp .env.example .env
```

Edit `.env` — the defaults work out of the box for mini:

```bash
KNOWLEDGEHUB_PROFILE=mini
DATABASE_URL=sqlite+aiosqlite:///./data/sqlite/knowledgehub.db
LLM_BACKEND=ollama
OLLAMA_MODEL=phi3:mini
```

### 2. Start the stack

```bash
make up
```

This starts: Gateway (8000), Admin (8001), Ollama (11434), ChromaDB (8100), Open WebUI (3000).

### 3. Pull an LLM model

```bash
make pull-model MODEL=phi3:mini
```

### 4. Initialize the database

```bash
make shell SERVICE=gateway
python scripts/init_db.py --seed
exit
```

### 5. Verify

```bash
# Health check
curl http://localhost:8000/health

# Open the chat UI
open http://localhost:3000

# Open the admin dashboard
open http://localhost:8001
```

---

## Enterprise Profile Setup

### 1. Prerequisites check

```bash
# Verify NVIDIA GPU
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` for enterprise:

```bash
KNOWLEDGEHUB_PROFILE=full

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://knowledgehub:your-secure-password@postgres:5432/knowledgehub
POSTGRES_DB=knowledgehub
POSTGRES_USER=knowledgehub
POSTGRES_PASSWORD=your-secure-password

# Qdrant
VECTORSTORE_BACKEND=qdrant

# vLLM with GPU
LLM_BACKEND=vllm
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_GPU_MEMORY=0.9

# Security
SECRET_KEY=$(openssl rand -hex 32)
ADMIN_API_KEY=$(openssl rand -hex 16)
```

### 3. Start the enterprise stack

```bash
make up-enterprise
```

This starts all mini services plus: PostgreSQL, Qdrant, Redis, vLLM (GPU).

### 4. Initialize database and seed

```bash
make shell SERVICE=gateway
python scripts/init_db.py --seed
python scripts/migrate.py upgrade
exit
```

### 5. Verify all services

```bash
python scripts/health_check.py --verbose
```

---

## Development Setup (without Docker)

### 1. Install Python 3.11+ and Poetry

```bash
# macOS
brew install python@3.11 poetry

# Ubuntu/Debian
sudo apt install python3.11 python3.11-venv
pip install poetry
```

### 2. Install dependencies

```bash
cd knowledgehub
poetry install
```

### 3. Start external services

Start Ollama separately:

```bash
ollama serve &
ollama pull phi3:mini
```

### 4. Run the gateway

```bash
make dev
```

### 5. Run the admin dashboard

```bash
make dev-admin
```

---

## Docker Build from Source

If you need custom Docker images:

```bash
# Build production image
docker compose build gateway

# Build dev image
docker compose -f docker-compose.yml -f docker-compose.dev.yml build

# Verify image size (target: < 500 MB)
docker images | grep knowledgehub
```

---

## Troubleshooting Installation

### Docker Compose version error

```
ERROR: Version in "./docker-compose.yml" is unsupported
```

**Fix:** Upgrade to Docker Compose v2:

```bash
docker compose version  # Must be v2.x+
```

### Ollama model not found

```
Error: model 'phi3:mini' not found
```

**Fix:** Pull the model first:

```bash
make pull-model MODEL=phi3:mini
```

### SQLite permission error

```
OperationalError: unable to open database file
```

**Fix:** Ensure data directory exists and is writable:

```bash
mkdir -p data/sqlite
chmod 777 data/sqlite
```

### GPU not detected (enterprise)

```
could not select device driver "nvidia"
```

**Fix:** Install nvidia-container-toolkit:

```bash
# Ubuntu/Debian
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

### Port already in use

```
Bind for 0.0.0.0:8000 failed: port is already allocated
```

**Fix:** Change the port in `.env`:

```bash
GATEWAY_PORT=8080
```

---

## Next Steps

- [Configuration](CONFIGURATION.md) — tune performance and security
- [Admin Guide](ADMIN_GUIDE.md) — set up detection rules and knowledge
- [User Guide](USER_GUIDE.md) — start using the chatbot
