
# Portfolio Optimization with Machine Learning

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)  [![React](https://img.shields.io/badge/react-18.x-brightgreen.svg)](https://react.dev)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Predict future returns & risk with ML, cluster similar assets, and build an optimal investing portfolio—served through a full‑stack web app.

---

## 📑 Table of Contents

1. [Features](#-features)
2. [Tech Stack](#-tech-stack)
3. [System Architecture](#-system-architecture)
4. [Project Structure](#-project-structure)
5. [Quick Start](#-quick-start)
6. [Detailed Setup](#-detailed-setup)
7. [Machine Learning Pipeline](#-machine-learning-pipeline)
8. [Front‑End Dashboard](#-front-end-dashboard)
9. [Deployment Guide](#-deployment-guide)
10. [Security](#-security)
11. [Data Sources](#-data-sources)
12. [TODO](#-todo)
13. [Contributing](#-contributing)
14. [License](#-license)
15. [Acknowledgements](#-acknowledgements)

---

## ✨ Features

* **ML predictions** (return & volatility) using regression models
* **K‑Means clustering** to enhance diversification
* **Mean–Variance / Sharpe‑ratio optimizer** via PyPortfolioOpt & SciPy
* **Interactive React dashboard** with Plotly/Recharts visualisations
* **User authentication** (AWS Cognito or Firebase Auth)
* **RESTful API** (FastAPI) secured with JWT
* **CI/CD** pipeline for auto‑deploy to AWS Amplify or Firebase Hosting

---

## ⚙️ Tech Stack

| Layer       | Technology                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------- |
| Front‑End   | React 18, Vite, TypeScript, TailwindCSS, Plotly.js                                          |
| Back‑End    | Python 3.11, FastAPI, Uvicorn, scikit‑learn, PyPortfolioOpt                                 |
| ML / Data   | pandas, numpy, yfinance, Charles Schwab API, Alpha Vantage                                  |
| Auth        | AWS Cognito (Amplify) **or** Firebase Auth                                                  |
| Infra       | Docker, GitHub Actions, AWS Amplify + ECS/Fargate **or** Firebase Hosting + Cloud Functions |
| Testing     | pytest, React Testing Library + Jest                                                        |
| Lint/CI     | ruff, black, eslint, prettier                                                               |

---

## 📐 System Architecture

```
 User ↔️ React (HTTPS) ↔️ API Gateway ↔️ FastAPI Server ↔️ ML Models & Optimizer ↔️ Data APIs
                                             ↘︎ RDS / S3 (cached data)
```

---

## 📂 Project Structure

```
root/
├── backend/
│   ├── app/                 # FastAPI code
│   ├── models/              # saved .pkl files
│   ├── data/                # cached CSVs & cluster info
│   └── Dockerfile
├── frontend/
│   ├── src/
│   ├── public/
│   └── Dockerfile
├── infra/                   # IaC (Terraform / CDK)
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1 Clone repo
$ git clone https://github.com/<your‑org>/portfolio-optimizer-ml.git
$ cd portfolio-optimizer-ml

# 2 Configure environment variables
$ cp .env.example .env
#   edit .env with API keys (SCHWAB_CLIENT_ID, SCHWAB_SECRET, AV_KEY, etc.)

# 3 Launch dev stack (requires Docker & Docker Compose)
$ docker compose up --build
# Front‑end → http://localhost:5173
# API       → http://localhost:8000
```

---

## 🛠️ Detailed Setup

### Backend

```bash
cd backend
poetry install   # or: pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Tests

```bash
pytest          # back‑end
npm test        # front‑end
```

---

## 🧠 Machine Learning Pipeline

1. **Data ingestion** – `scripts/fetch_data.py` downloads historical OHLC prices via Schwab / yfinance.
2. **Feature engineering** – compute log‑returns, rolling volatility, momentum.
3. **Clustering** – K‑Means (scikit‑learn) → cluster labels saved to `data/clusters.csv`.
4. **Prediction** – train regressors (RandomForest, Linear) to forecast next‑period return & volatility → save models to `models/`.
5. **Optimization** – Efficient Frontier or max‑Sharpe solved at request time using predictions & cov matrix.
6. **Serving** – `/optimize` endpoint returns weights & metrics as JSON.

---

## 📊 Front‑End Dashboard

* **Risk tolerance slider** (0–100)
* **Run Optimization** button → calls API
* **Allocation pie chart** & KPI cards
* **Back‑test line chart** vs S\&P 500
* **Cluster heatmap** of assets

---

## ☁️ Deployment Guide

### AWS Amplify Workflow

1. Push repo to GitHub.
2. In AWS Amplify Console → "Create App" → connect repo.
3. Add *Amplify Hosting* + *Cognito auth* backend environment.
4. Configure build settings (`amplify.yml`) to build React & deploy.
5. Back‑end container (`backend/Dockerfile`) deployed on ECS/Fargate → set API\_URL env in Amplify.

### Firebase Alternative

1. `firebase init` → Hosting & Functions.
2. Deploy FastAPI via Cloud Run (`gcloud run deploy`).
3. Update `.env.production` with public API URL & Firebase config.
4. `firebase deploy` for front‑end hosting & auth.

---

## 🔐 Security

* Secrets stored in AWS SSM Parameter Store or Firebase Config.
* HTTPS enforced via CloudFront or Firebase CDN.
* JWT access tokens required on all API routes.
* CORS restricted to whitelisted domains.

---

## 🗂️ Data Sources

| Source             | Purpose                  | Notes                                   |
| ------------------ | ------------------------ | --------------------------------------- |
| Charles Schwab API | Live & historical prices | Requires brokerage sandbox & OAuth flow |
| yfinance           | Historical daily OHLC    | Free & quick for prototyping            |
| Alpha Vantage      | Intraday data            | Free tier (5 req/min)                   |

---

## 📈 TODO

* [ ] Monte‑Carlo stress testing
* [ ] Scheduled rebalancing (cron)
* [ ] Support bonds, ETFs, crypto
* [ ] Dark / light theme

---

## 🧑‍💻 Contributing

1. Fork → create feature branch (`git checkout -b feat/my-feature`)
2. Commit & push with descriptive messages (lint passes)
3. Open Pull Request → ensure CI passes → get review 👍

---

## 📝 License

MIT © 2025 David Huang & Contributors

---

## 🙏 Acknowledgements

* *Modern Portfolio Theory* — Harry Markowitz (1952)
* Data from Charles Schwab, Yahoo Finance
* Charts powered by Plotly.js

> “Risk comes from not knowing what you’re doing.” — *Warren Buffett*
