# Prognostix ML

Scalable predictive maintenance system featuring advanced deep learning models, real-time monitoring, and API serving for industrial reliability.

## 📌 Overview

Prognostix ML is an end-to-end Machine Learning platform designed to predict the **Remaining Useful Life (RUL)** of industrial equipment. Built around the CMAPSS (Turbofan Engine Degradation) dataset, it features a complete lifecycle from data ingestion and feature engineering to model training, API serving, and continuous performance monitoring.

## 🏗️ Architecture

- **Deep Learning Models (`src/models/`)**: Includes state-of-the-art architectures such as LSTMs, GRUs, and Attention mechanisms, alongside Random Forest baselines.
- **Data Pipeline (`src/features/`, `scripts/`)**: Automated data preparation, scaling, and generation of rolling/lag window features.
- **Inference API (`api/`)**: High-performance backend (FastAPI) for serving real-time predictions.
- **Monitoring (`monitoring/`)**: Built-in scripts to track model drift, performance degradation, and automatic alert generation.
- **Dockerized Deployment (`docker/`)**: Fully containerized API and Frontend via `docker-compose`.

## ⚙️ Getting Started

### Prerequisites

You can run this project using either **Docker** (recommended) or a local Python environment.

### 1. Using Docker (Recommended)

Use the provided `docker-compose` setup to spin up the entire stack (API + Frontend):

```bash
cd docker
docker-compose up --build
```

### 2. Local Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/HasnatKhan010/prognostix-ml.git
cd prognostix-ml

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Training Models

To train a new model from scratch, run the end-to-end training pipeline:

```bash
# 1. Process data and engineer features
python scripts/prepare_data.py

# 2. Train a specific model (e.g., lstm, gru, attention, baseline)
python scripts/train.py --model lstm

# 3. Evaluate on the test set
python scripts/evaluate.py
```

### Running the API locally

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
*Interactive API documentation (Swagger UI) will be automatically available at `http://localhost:8000/docs`*

## 🧪 Testing

Run the automated test suite to verify data preprocessing, feature engineering, and API functionality:

```bash
pytest tests/
```

## 📈 Monitoring

To execute data drift analysis and model performance checks on new incoming telemetry data:

```bash
python scripts/predict.py
python -m monitoring.drift
```
