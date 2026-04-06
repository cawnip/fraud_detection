---
title: Credit Card Fraud Detection
emoji: 💳
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10
app_file: app.py
pinned: false
---

# Credit Card Fraud Detection

Kredi kartı işlemlerini gerçek zamanlı analiz eden, makine öğrenmesi tabanlı fraud tespiti sistemi.

284.807 işlem içeren gerçek banka verisinde **%0.17 fraud oranı** (ciddi class imbalance) ile çalışır. SMOTE, Optuna hyperparameter tuning, SHAP açıklanabilirlik ve F2-score threshold optimizasyonu içerir.

**Demo:** *(HF Spaces deploy sonrası eklenecek)*

---

## Teknik Özellikler

| Özellik | Detay |
|--------|-------|
| Model | XGBoost (Optuna ile tuned) |
| Class imbalance | SMOTE oversampling |
| Threshold | F2-score (recall ağırlıklı) ile optimize |
| Cross-validation | Stratified K-Fold (5 fold) |
| Açıklanabilirlik | SHAP TreeExplainer |
| ROC-AUC | 0.975 |
| API | FastAPI + Pydantic v2 |
| UI | Gradio |

---

## Pipeline

```
creditcard.csv
    │
    ├── EDA (class distribution, amount distribution)
    │
    ├── Preprocessing
    │   ├── StandardScaler → Amount, Time
    │   └── Stratified train/test split (%80/%20)
    │
    ├── SMOTE (training set'te minority class oversample)
    │
    ├── Hyperparameter Tuning
    │   └── Optuna — 30 trial, PR-AUC maximize, StratifiedKFold içinde
    │
    ├── Model Eğitimi
    │   ├── Logistic Regression (baseline)
    │   ├── Random Forest
    │   └── XGBoost (tuned) ← deploy edilen
    │
    ├── Threshold Optimizasyonu
    │   └── PR curve üzerinde F2-score (beta=2) maximize
    │
    ├── Değerlendirme
    │   ├── ROC-AUC, PR-AUC, F1, MCC, Precision, Recall
    │   └── Confusion matrix, ROC curve, feature importance
    │
    └── SHAP
        └── TreeExplainer — global + local açıklanabilirlik
```

---

## Teknik Seçimler

**Neden XGBoost?**
Gradient boosting, fraud tespiti gibi imbalanced ve tablüler veri problemlerinde random forest ve lojistik regresyona kıyasla tipik olarak daha yüksek PR-AUC üretir. Eğitim süresi ve model boyutu da deployment için uygundur (≈350KB).

**Neden F2-score ile threshold?**
Fraud tespitinde yanlış negatif (kaçırılan dolandırıcılık) yanlış pozitiften çok daha maliyetlidir. F2-score recall'a daha fazla ağırlık verdiği için varsayılan 0.5 threshold yerine recall-odaklı optimal eşik kullanıldı.

**Neden SMOTE?**
%0.17 fraud oranında model çoğunluk sınıfına yönelir. SMOTE yalnızca training set'e uygulanır — test seti gerçek dağılımı korur.

**Neden PR-AUC (Optuna objective)?**
İmbalanced veri setlerinde ROC-AUC yanıltıcı olabilir. PR-AUC minority sınıfı (fraud) üzerindeki performansı daha hassas ölçer.

---

## Proje Yapısı

```
fraud-ai-system/
├── data/                   # creditcard.csv buraya gelecek (git'e dahil değil)
├── models/                 # Eğitilmiş modeller (.joblib)
├── logs/                   # Eğitim logları
├── plots/                  # EDA ve değerlendirme görselleri
├── src/
│   ├── data_loader.py      # CSV yükleme, temel istatistikler
│   ├── preprocessing.py    # Scaling, split, SMOTE
│   ├── model.py            # Model tanımları
│   ├── trainer.py          # Eğitim, kaydetme, yükleme
│   ├── evaluator.py        # Metrik hesaplama, threshold kaydetme
│   ├── cross_validation.py # StratifiedKFold CV
│   ├── tuner.py            # Optuna hyperparameter tuning
│   ├── explainer.py        # SHAP hesaplama ve görseller
│   └── visualization.py    # EDA ve sonuç grafikleri
├── utils/
│   └── metrics.py          # find_optimal_threshold, compute_metrics
├── main.py                 # Tam eğitim pipeline'ı
├── app.py                  # FastAPI + Gradio deployment
├── config.py               # Merkezi konfigürasyon
├── requirements.txt        # HF deploy (minimal bağımlılıklar)
└── requirements-train.txt  # Eğitim/EDA bağımlılıkları
```

---

## Kurulum

```bash
git clone <repo-url>
cd fraud-ai-system

# HF deploy / inference için
pip install -r requirements.txt

# Model eğitimi için (opsiyonel, yerelde training çalıştıracaksan)
pip install -r requirements-train.txt
```

> **macOS:** XGBoost için `libomp` gereklidir:
> ```bash
> brew install libomp
> ```

Kaggle'dan [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) indirip `data/` klasörüne koyun.

---

## Kullanım

### Model Eğitimi

```bash
python main.py
```

Eğitim tamamlandığında `models/` altında `.joblib` dosyaları ve `plots/` altında görseller oluşur.

### Web Arayüzü

```bash
python app.py
```

`http://localhost:7860` adresinden Gradio arayüzüne, `/predict` endpoint'inden FastAPI'ye erişilebilir.

### API

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [406.0, -2.31, 1.95, ...]}'
```

```json
{
  "prediction": 1,
  "fraud_probability": 0.9923,
  "threshold": 0.9878
}
```

---

## Değerlendirme Metrikleri

- **ROC-AUC:** 0.975
- **Threshold:** F2-score ile optimize (varsayılan 0.5 yerine)
- **Metrikler:** ROC-AUC, PR-AUC, F1, Precision, Recall, MCC

---

## Veri Seti

[ULB Machine Learning Group — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284.807 işlem, 492 fraud (%0.17)
- V1–V28: PCA ile anonimleştirilmiş banka özellikleri
- Amount, Time: orijinal değerler (ölçeklendi)
- Class: 0 = normal, 1 = fraud
