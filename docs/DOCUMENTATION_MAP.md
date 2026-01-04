# 📚 Documentation Map

Welcome to the **EUR/USD Capstone Project** documentation. This guide helps you navigate the technical details of the project.

## 🏗️ Architecture
Deep Dives into how the system is built.
- **[Storage Architecture](architecture/4_STORAGE_ARCHITECTURE.md)**: Detailed breakdown of S3, RDS, and Local storage tiers.
- **[Data Pipeline Architecture](architecture/DATA_PIPELINE_ARCHITECTURE.md)**: How data flows from ingestion to training.
- **[Model Selection Flow](architecture/MODEL_SELECTION_FLOW.md)**: Logic for choosing the "Champion" model.
- **[Data Flow for Predictions](architecture/DATA_FLOW_FOR_PREDICTIONS.md)**: How live inference works.
- **[Storage Comparison](architecture/3_STORAGE_COMPARISON.md)**: Analysis of why we chose this hybrid storage approach.

## 🚀 Deployment
Everything related to infrastructure and deploying to AWS.
- **[AWS Deployment Guide](deployment/6_AWS_DEPLOYMENT_GUIDE.md)**: Main guide for deploying the full stack.
- **[Docker Quickstart](deployment/5_DOCKER_QUICKSTART.md)**: Working with the Docker containers.
- **[Deployment Checklist](deployment/8_DEPLOYMENT_CHECKLIST.md)**: Steps to ensure a production-ready deploy.
- **[EC2 Setup & SSH](deployment/7_EC2_SETUP_AND_SSH.md)**: Managing the compute instances.
- **[Deploy Scripts Explained](deployment/9_DEPLOY_SCRIPT_EXPLAINED.md)**: Reference for the automation scripts in `scripts/`.
- **[MLflow on AWS](deployment/10_MLFLOW_AWS_DEPLOYMENT.md)**: Specifics of the MLflow Tracking Server deployment.

## 📘 Guides & Manuals
Usage instructions for maintenance and operations.
- **[Retraining Manual](guides/RETRAINING_MANUAL.md)**: How the automated daily retraining works and how to monitor it.
- **[MLflow Setup Guide](guides/11_MLFLOW_SETUP_GUIDE.md)**: Configuring the MLflow experiment tracker.
- **[FAQ / Troubleshooting](guides/faq_guide.md)**: Common issues and fixes (e.g., date parsing, S3 permissions).

## 🐛 Debugging
Notes and logs for troubleshooting.
- **[Debug Summary](debug/debug_summary_flask_deployment.md)**: Logs and resolution summary for previous Flask deployment issues.

---
*Return to [Main README](../README.md)*
