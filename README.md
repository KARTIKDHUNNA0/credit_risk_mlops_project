# 💳 Minimal Credit Risk MLOps Pipeline

![CI/CD Status](https://github.com/KARTIKDHUNNA0/credit_risk_mlops_project/actions/workflows/ci.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11-blue)
![Prefect](https://img.shields.io/badge/Orchestration-Prefect-orange)
![FastAPI](https://img.shields.io/badge/Serving-FastAPI-green)

**A proof-of-concept MLOps workflow for credit default prediction.**

This project isn't trying to be a massive enterprise system. Instead, it is a **minimal, functional implementation** designed to explore how modern MLOps tools (Prefect, DVC, FastAPI, GitHub Actions) connect in the real world. It takes the raw Home Credit dataset and moves it through a reproducible pipeline—from ingestion to a live API endpoint.

---

## 🏗️ System Architecture

The system follows a standard modular pattern to separate training logic from serving logic:

```mermaid
graph LR
    subgraph Data_Pipeline [Prefect Orchestration]
        A("Raw Data (DVC)") --> B(Ingest Flow)
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
