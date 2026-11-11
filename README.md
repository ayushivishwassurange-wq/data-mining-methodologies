# Principled Data Science Across Three Methodologies: CRISP-DM, SEMMA, and KDD

**Author:** Ayushi Vishwas Surange  
**Student ID:** 020795684  
**Course:** CMPE 256 — Data Mining  
**Assignment:** Data Mining Process Assignment (CRISP-DM, SEMMA, and KDD Methodologies)  

---

## Submission Links — Video Walkthrough, Medium Articles & Google Colab Notebooks

- **YouTube Code & Artifact Walkthrough Video:** [https://youtu.be/wb7e6PHCI2o](https://youtu.be/wb7e6PHCI2o)

| Methodology | Published Medium.com Article | Google Colab Notebook | Local Notebook Path |
| :--- | :--- | :--- | :--- |
| **1. CRISP-DM** | [**Beyond Accuracy: Unlocking $271K in Telco Retention ROI with CRISP-DM**](https://medium.com/@ayushivishwas.surange/beyond-accuracy-how-we-used-crisp-dm-and-an-ai-critic-persona-to-unlock-271k-in-telco-retention-e0da11fb715b) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayushivishwassurange-wq/data-mining-methodologies/blob/main/CRISP-DM_Telco_Customer_Churn/CRISP_DM_Telco_Churn.ipynb) | [`CRISP-DM_Telco_Customer_Churn/CRISP_DM_Telco_Churn.ipynb`](CRISP-DM_Telco_Customer_Churn/CRISP_DM_Telco_Churn.ipynb) |
| **2. SEMMA** | [**Inside SAS SEMMA: Stratified 3-Way Sampling, Intensive Ratios & Decile Lift**](https://medium.com/@ayushivishwas.surange/inside-sas-semma-how-stratified-3-way-sampling-intensive-ratios-and-decile-lift-transformed-1f4813bf56bd) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayushivishwassurange-wq/data-mining-methodologies/blob/main/SEMMA_California_Housing_Valuation/SEMMA_California_Housing.ipynb) | [`SEMMA_California_Housing_Valuation/SEMMA_California_Housing.ipynb`](SEMMA_California_Housing_Valuation/SEMMA_California_Housing.ipynb) |
| **3. KDD** | [**Data Mining Is Just One Step: Full KDD Process for Clinical Rules & Phenotypes**](https://medium.com/@ayushivishwas.surange/data-mining-is-just-one-step-using-the-full-kdd-process-to-extract-clinical-rules-patient-1a7b13bfa8ed) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayushivishwassurange-wq/data-mining-methodologies/blob/main/KDD_Heart_Disease_Knowledge_Discovery/KDD_Heart_Disease_Discovery.ipynb) | [`KDD_Heart_Disease_Knowledge_Discovery/KDD_Heart_Disease_Discovery.ipynb`](KDD_Heart_Disease_Knowledge_Discovery/KDD_Heart_Disease_Discovery.ipynb) |

---

## Executive Overview

This repository demonstrates **principled, phase-by-phase, end-to-end data science** across the three foundational data mining methodologies—**CRISP-DM**, **SEMMA**, and **KDD**—on **three distinct real-world domain datasets**. Every phase, stage, and subtask was iteratively audited and refined through multiple rounds of review using a **"World-Renowned Authority & Award-Winning Textbook Author" AI Critic Persona**.

| Methodology | Subdirectory | Domain & Dataset (`N` Rows) | Stages / Subtasks Completed | Champion Model & Key Metric | Core Business / Scientific Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. CRISP-DM** | [`CRISP-DM_Telco_Customer_Churn/`](CRISP-DM_Telco_Customer_Churn/) | **Telecommunications Subscriber Churn** (`7,043` rows, `16` features) | **All 6 Phases & 24 Subtasks** (Business Understanding $\rightarrow$ Deployment) | `GradientBoostingClassifier` (`Test ROC-AUC = 0.8580`, `Brier = 0.1288`) | Shifting threshold from `0.50` to **`p* = 0.29`** yields **`$54,372.50`** net profit on test cohort (**`+$10,075` / `+22.7%` ROI gain**) |
| **2. SEMMA** | [`SEMMA_California_Housing_Valuation/`](SEMMA_California_Housing_Valuation/) | **California Real Estate Valuation** (`10,000` rows, `10` features) | **All 5 SAS SEMMA Stages** (Sample, Explore, Modify, Model, Assess) | `HistGradientBoostingRegressor` (`Test R2 = 0.8642`, `RMSE = $38,790`) | Stratified 60/20/20 split (`KS p = 0.8912`), intensive ratios, and **10-Decile Lift Curve** achieving **`1.517x` Top-Decile Lift** |
| **3. KDD** | [`KDD_Heart_Disease_Knowledge_Discovery/`](KDD_Heart_Disease_Knowledge_Discovery/) | **Clinical Cardiovascular Disease** (`1,500` rows, `17` raw attributes) | **All 5 Fayyad KDD Steps** (Selection, Preprocessing, Transformation, Data Mining, Interpretation) | `GradientBoostingClassifier` (`ROC-AUC = 0.9124`, `Sensitivity = 87.41%`) + `RuleTree` | Discovered **3 clinical patient phenotypes** (`16.4%`, `49.2%`, `86.7%` prevalence) & **8 symbolic `IF-THEN` rules** (up to **`95.8%` confidence**) |

---

## Repository Structure

```text
data-mining-methodologies/
├── README.md                                       # Master portfolio & comparative methodology guide
├── build_assignment.py                             # Reproducible master generator script
├── execution_summary.json                          # Complete JSON metrics & audit trail across all 3 projects
│
├── CRISP-DM_Telco_Customer_Churn/                  # PROJECT 1: CRISP-DM (6 Phases, 24 Subtasks)
│   ├── CRISP_DM_Telco_Churn.ipynb                  # Executed Google Colab Notebook (with embedded PNG plots)
│   ├── critic_revisions.md                         # World-Renowned CRISP-DM Critic Persona multi-revision log
│   ├── detailed_report.md                          # Full technical report covering all 6 phases & 24 subtasks
│   ├── medium_article.md                           # Medium.com publication article
│   ├── data/telco_customer_churn.csv               # Dataset (7,043 rows x 16 columns)
│   ├── figures/                                    # High-resolution phase visualizations (.png)
│   ├── models/crisp_dm_churn_pipeline.pkl          # Serialized champion pipeline & threshold artifact
│   └── deployment/                                 # Production Streamlit ROI app + PSI Drift Monitor + Dockerfile
│
├── SEMMA_California_Housing_Valuation/             # PROJECT 2: SAS SEMMA (5 Stages)
│   ├── SEMMA_California_Housing.ipynb              # Executed Google Colab Notebook (with embedded PNG plots)
│   ├── critic_revisions.md                         # World-Renowned SAS SEMMA Critic Persona multi-revision log
│   ├── detailed_report.md                          # Full technical report covering Sample, Explore, Modify, Model, Assess
│   ├── medium_article.md                           # Medium.com publication article
│   ├── data/california_housing_semma.csv           # Dataset (10,000 rows x 10 columns)
│   ├── figures/                                    # High-resolution stage visualizations (.png)
│   ├── models/semma_housing_pipeline.pkl           # Serialized regression pipeline & decile lift table
│   └── deployment/app.py                           # Property valuation & Decile Lift tier scorer
│
└── KDD_Heart_Disease_Knowledge_Discovery/          # PROJECT 3: KDD PROCESS (5 Steps)
    ├── KDD_Heart_Disease_Discovery.ipynb           # Executed Google Colab Notebook (with embedded PNG plots)
    ├── critic_revisions.md                         # World-Renowned KDD Pioneer Critic Persona multi-revision log
    ├── detailed_report.md                          # Full technical report covering Selection -> Interpretation
    ├── medium_article.md                           # Medium.com publication article
    ├── data/clinical_heart_disease_kdd.csv         # Dataset (1,500 rows x 17 columns)
    ├── figures/                                    # High-resolution step visualizations (.png)
    ├── models/extracted_clinical_rules.txt         # Explicit human-readable IF-THEN clinical decision rules
    ├── models/kdd_heart_disease_models.pkl         # Serialized models, rules, and phenotype clusters
    └── deployment/app.py                           # Bedside Clinical Decision Support & Rule Explainer
```

---

## Comparative Analysis: When to Use CRISP-DM vs. SEMMA vs. KDD

| Dimension | CRISP-DM | SEMMA (SAS Institute) | KDD Process (Fayyad et al.) |
| :--- | :--- | :--- | :--- |
| **Primary Orientation** | **Business & Lifecycle Governance** | **Statistical & Algorithmic Modeling** | **Scientific & Interpretable Knowledge Extraction** |
| **Starting Point** | Phase 1: Business Understanding (Cost-Benefit Matrix, KPIs, Risks) | Stage 1: Sample (Stratified Train / Validation / Test Partitioning) | Step 1: Selection (Isolating relevant domain records/variables from raw DBs) |
| **Ending Point** | Phase 6: Deployment & Drift Monitoring (PSI, MLOps, SLA) | Stage 5: Assess (Holdout Decile Lift Charts, Gains Curves, Residuals) | Step 5: Interpretation / Evaluation (Symbolic rules, phenotypes, human validation) |
| **Best Suited For** | Enterprise product features, churn prevention, fraud ROI, MLOps | High-volume tabular scoring, credit risk, real estate valuation, marketing lift | Healthcare diagnostics, scientific discovery, regulatory/explainable AI |
