# 💳 Credit Risk MLOps Project (Home Credit Default Risk)

![CI/CD Status](https://github.com/KARTIKDHUNNA0/credit_risk_mlops_project/actions/workflows/ci.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11-blue)
![Prefect](https://img.shields.io/badge/Orchestration-Prefect-orange)
![FastAPI](https://img.shields.io/badge/Serving-FastAPI-green)

A production-ready MLOps pipeline for predicting credit default risk. This project implements an end-to-end workflow from data ingestion to model serving, emphasizing reproducibility, automation, and data versioning.

---

## 🏗️ System Architecture

The system follows a modular MLOps design pattern:

```mermaid
graph LR
    subgraph Data_Pipeline [Prefect Orchestration]
        A[Raw Data (DVC)] --> B(Ingest Flow)
        B --> C(Feature Engineering Flow)
        C --> D(Training Flow / LightGBM)
        D --> E{Model Artifact}
    end
    
    subgraph Serving [FastAPI Inference]
        E --> F[API Server]
        G[User / Client] -- POST /predict --> F
        F -- Prediction --> G
    end
    
    subgraph CI_CD [GitHub Actions]
        H[Push to Main] --> I[Install Deps]
        I --> J[Run Tests & Dry Run]
    end
