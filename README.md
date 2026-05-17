# 🌾 ML-Model on Crop Classification & Recommendation System Using Satellite Data

> **B.Tech Project (2025–2026) | IIT Delhi | Department of Mathematics**

An end-to-end agricultural intelligence framework combining **satellite remote sensing**, **machine learning**, and **deep learning** to classify crops, forecast weather, predict market prices, and propose data-driven price stabilization strategies — with a focused case study on **onion** in Nashik, Maharashtra.

---

## 👥 Team

| Name | Roll No |
|---|---|
| Rahul Shah | 2022MT11732 |
| Harsh Chaudhary | — |

**Supervisor:** Prof. Rahul Singh, Department of Mathematics, IIT Delhi

---

## 📋 Table of Contents

- [Overview](#overview)
- [Three Pipelines](#three-pipelines)
- [Pipeline 1: Crop Classification](#pipeline-1-crop-classification)
- [Pipeline 2: LSTM Price Forecasting](#pipeline-2-lstm-price-forecasting)
- [Pipeline 3: Price Stabilization & Market Analysis](#pipeline-3-price-stabilization--market-analysis)
- [Datasets](#datasets)
- [Results](#results)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [Tech Stack](#tech-stack)

---

## 🔭 Overview

Agriculture in India faces mounting challenges from climate variability, resource constraints, and volatile market prices. This project addresses these challenges through three interconnected data pipelines:

1. **Crop Classification** — Identify and map crop types spatially using multi-temporal satellite imagery
2. **Price Forecasting** — Predict onion market prices using LSTM deep learning on historical mandi data
3. **Price Stabilization** — Analyze market dynamics and propose a storage-based strategy to reduce farmer distress

The framework targets two study regions:
- **Uttar Pradesh** — General crop classification across ~15 crop types
- **Nashik, Maharashtra** — Focused onion classification, price analysis, and market study

---

## 🔁 Three Pipelines

```
┌──────────────────────────────────────────────────────────────────┐
│              PIPELINE 1: CROP CLASSIFICATION                     │
│  Sentinel-1/2 → Feature Engineering → Random Forest → Crop Map  │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│           PIPELINE 2: LSTM PRICE FORECASTING                     │
│  Mandi Price Data → Preprocessing → LSTM → 30-Day Forecast       │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│       PIPELINE 3: PRICE STABILIZATION & MARKET ANALYSIS          │
│  EDA + Weather Correlation → Storage Strategy → Policy Insights  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛰️ Pipeline 1: Crop Classification

### 1A — General Crop Classification (Uttar Pradesh)

- **Platform:** Google Earth Engine (GEE)
- **Data:** Multi-temporal Sentinel-2 surface reflectance imagery (July 2022 – Nov 2023)
- **Spectral Bands:** B2, B3, B4, B5, B6, B8, B9, B11, B12
- **Vegetation Indices:**

$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$

$$\text{SAVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + L}(1 + L), \quad L = 0.5$$

- **Ground Truth:** Manually labeled via Google Earth & Street View
- **Classifier:** Random Forest (100 trees), 70/30 train-test split
- **Output:** Spatially explicit map covering ~15 crop classes

### 1B — Onion Crop Classification (Nashik)

- **Seasons:** 6 agricultural cycles (Oct 2020 – Mar 2026), each with 6 monthly observations
- **Satellite Data:** Sentinel-2 (optical) + Sentinel-1 SAR (VV, VH polarizations)
- **Features engineered:**
  - Vegetation indices: NDVI, NDRE, GCVI, MOIS, BSI, NDTI, Chlorophyll Gap
  - Temporal: monthly values + inter-month differences (Δ features)
  - Ratio features: NDVI/BSI, NDRE/MOIS
  - Phenological: seasonal mean/max NDVI, NDRE
  - SAR: VH backscatter, VH/VV ratio, peak-season response, texture
  - Spatial: latitude
- **Ground Truth:** 320 manually labeled points (onion / non-onion) via Google Street View
- **Classifier:** Random Forest (150 trees, min leaf population: 15, bagging fraction: 0.5)
- **Post-processing:** Probability threshold = 0.46; focal filter smoothing
- **Validation:** Leave-One-Season-Out (LOSO) cross-validation
- **Output:** Season-wise binary onion crop masks at 10 m resolution

---

## 📈 Pipeline 2: LSTM Price Forecasting

- **Data:** Daily onion mandi price records from Nashik (2020–2025) via [Data.gov.in](https://data.gov.in)
- **Target variable:** Modal price (best representation of prevailing market price)
- **Preprocessing:**
  - Missing value interpolation
  - Outlier removal via statistical thresholds
  - Min-Max normalization: $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$
- **Sequence format (sliding window):**
$$\{x_{t-T}, x_{t-T+1}, \ldots, x_{t-1}\} \rightarrow x_t$$
- **Model Architecture:**
  - LSTM layers (temporal pattern capture)
  - Dropout layers (regularization)
  - Dense output layer
  - Optimizer: Adam | Loss: Mean Squared Error (MSE)
- **Output:** Short-to-medium term price forecasts including 30-day forward projections

---

## 📊 Pipeline 3: Price Stabilization & Market Analysis

- **Data:** 24,000+ records across 45 Nashik markets (2020–2025)
- **EDA focused on:**
  - Temporal trends (daily, monthly, yearly)
  - Seasonal peaks and troughs identification
  - Inter-annual volatility patterns
  - Price–weather correlation (temperature, humidity, soil moisture, precipitation)
- **Key observation:** Small average price spread (~₹75/quintal) between Nashik (producer) and Mumbai (consumer) markets — volatility originates at production level, not supply chain
- **Proposed Strategy:** Decentralized cold storage enabling farmers to:
  - Store produce during harvest (low-price) periods
  - Sell strategically during high-price periods
  - Reduce dependence on reactive government MSP/import-export interventions

---

## 🗃️ Datasets

| Dataset | Source | Coverage |
|---|---|---|
| Sentinel-2 Optical Imagery | Copernicus Open Access Hub | 2022–2026 |
| Sentinel-1 SAR (VV, VH) | Copernicus Open Access Hub | 2022–2026 |
| Meteorological Data | NASA (POWER / MERRA-2) | 2020–2026 |
| Onion Mandi Prices | Data.gov.in | 2020–2025 |
| Ground Truth Labels | Google Earth + Street View | Manual annotation |

---

## 📉 Results

### Crop Classification — Uttar Pradesh (Random Forest)

| Metric | Training | Testing |
|---|---|---|
| Overall Accuracy | 0.9989 | 0.9754 |
| Cohen's Kappa (κ) | 0.9942 | 0.8749 |

### Onion Classification — Nashik (LOSO Cross-Validation)

| Test Season | Confusion Matrix | OA | κ |
|---|---|---|---|
| Oct 2022 – Mar 2023 | TP=55, FP=13, FN=10, TN=29 | 0.7850 | 0.5435 |
| Oct 2023 – Mar 2024 | TP=46, FP=24, FN=12, TN=39 | 0.7025 | 0.4088 |
| Oct 2025 – Mar 2026 | TP=37, FP=11, FN=3, TN=21 | **0.8056** | **0.5962** |

### LSTM Onion Price Forecasting

| Metric | Value |
|---|---|
| MAE | 124.22 |
| RMSE | 164.45 |
| R² Score | 0.3452 |
| MAPE | 0.1123 (11.23%) |
| Accuracy | 88.77% |

---

## 💡 Key Findings

- **Cereal prices** (wheat, rice) remain relatively stable throughout the year and are more resilient to climate shocks.
- **Top 3 volatile vegetables** — Onion, Tomato, Potato — show extreme inter-annual price swings driven primarily by unseasonal rains, heatwaves, droughts, and floods.
- **Geographic price spread** is negligible across regions, indicating efficient and cheap transportation infrastructure — price volatility is localized to the production zone.
- **Weather correlations:** Soil moisture, humidity, and temperature show moderate correlation with onion prices; precipitation shows minimal direct influence.
- **Government interventions** (MSP, export bans, import policies) are reactive, costly, and often distort natural market dynamics — a proactive storage-based approach is more sustainable.

---

## 🔮 Future Work

- Extend crop classification to **Tomato and Potato** — India's other two highly volatile vegetables.
- Use the satellite classifier to **map production clusters** of the big 3 vegetables at national scale.
- Combine **weather event detection** (unseasonal rain, heatwave flags) with **production cluster maps** to build an early warning system for price spikes.
- Improve LSTM forecasting by incorporating **satellite-derived crop area estimates** as exogenous features.
- Evaluate the **economic viability** of decentralized cold storage — cost-benefit analysis for smallholder farmers.
- Explore **multi-market price forecasting** across other producing regions (e.g., Kurnool for onion, Nashik vs. Pune comparison).

---

## 🛠️ Tech Stack

| Category | Tools / Technologies |
|---|---|
| Geospatial Processing | Google Earth Engine (GEE), Python |
| Satellite Data | Sentinel-1 SAR, Sentinel-2 Optical (Copernicus) |
| Machine Learning | Random Forest, SVM (Scikit-learn) |
| Deep Learning | LSTM (TensorFlow / Keras) |
| Data Analysis | Pandas, NumPy, Matplotlib, Seaborn |
| Ground Truth Collection | Google Earth, Google Street View |
| Market Data | Data.gov.in |
| Meteorological Data | NASA POWER API |

---

## 📄 Citation

If you use this work, please cite:

```
Rahul Shah, Harsh Chaudhary (2026).
ML-Model on Crop Classification and Recommendation System Using Satellite Data on Indian Agriculture.
B.Tech Project Report, Indian Institute of Technology Delhi.
Supervised by Prof. Rahul Singh, Department of Mathematics.
```

---

## 📜 License

This project was developed for academic purposes as part of the B.Tech curriculum at IIT Delhi (2025–2026). All satellite data usage complies with the Copernicus Open Access data policy.
