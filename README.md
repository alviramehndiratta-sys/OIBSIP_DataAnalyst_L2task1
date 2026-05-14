# 🏠 House Price Prediction — Linear Regression

A clean, end-to-end machine learning project that predicts house prices using Linear Regression on the [Housing dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **R² Score** | 0.6495 |
| **RMSE** | ₹13,31,071 |
| **MAE** | ₹9,79,680 |
| Train samples | 436 |
| Test samples | 109 |
| Features | 12 |

👉 **[View interactive visualisation](outputs/visualisation.html)** — open in any browser.

---

## 📁 Project Structure

```
house-price-prediction/
├── data/
│   └── Housing.csv          # Raw dataset (545 rows × 13 columns)
├── outputs/
│   ├── visualisation.html   # Interactive charts dashboard
│   ├── model.pkl            # Trained model (joblib)
│   └── metrics.json         # Evaluation metrics
├── notebooks/
│   └── analysis.ipynb       # Step-by-step EDA + modelling notebook
├── train.py                 # Training script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone & enter
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train.py

# 4. Open the visualisation
open outputs/visualisation.html     # macOS
xdg-open outputs/visualisation.html # Linux
```

---

## 🔧 Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `area` | Numeric | Plot area in sq ft |
| `bedrooms` | Numeric | Number of bedrooms |
| `bathrooms` | Numeric | Number of bathrooms |
| `stories` | Numeric | Number of floors |
| `mainroad` | Binary | Faces a main road |
| `guestroom` | Binary | Has a guest room |
| `basement` | Binary | Has a basement |
| `hotwaterheating` | Binary | Hot-water heating system |
| `airconditioning` | Binary | Air conditioning |
| `parking` | Numeric | Parking spaces |
| `prefarea` | Binary | In a preferred area |
| `furnishingstatus` | Ordinal | 0 = unfurnished, 1 = semi, 2 = furnished |

---

## 📈 Key Findings

- **Bathrooms** is the single strongest predictor (+₹10.97L per extra bathroom)
- **Air conditioning** adds ~₹7.86L on its own
- **Hot water heating** and **preferred area** each add ~₹6–6.3L
- **Area (sq ft)** contributes ₹236 per sq ft — seemingly small, but meaningful at scale
- The model explains **~65% of price variance** — strong for a plain linear model with no polynomial features

---

## 🛠 Tech Stack

- Python 3.10+
- pandas · numpy · scikit-learn · joblib
- Vanilla HTML/CSS/JS + Chart.js (visualisation — zero dependencies to install)

---

## 💡 Potential Improvements

- [ ] Polynomial features for `area`
- [ ] Ridge / Lasso regularisation
- [ ] Gradient Boosting (XGBoost / LightGBM)
- [ ] Cross-validation (k-fold)
- [ ] Feature interaction terms

---

## 📄 License

MIT — free to use, modify, and distribute.
