import logging
import numpy as np
import joblib
import xgboost as xgb
import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
from config import MODEL_DIR, DEPLOY_MODEL, HOST, PORT

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_PATH = MODEL_DIR / f"{DEPLOY_MODEL}.joblib"
THRESHOLD_PATH = MODEL_DIR / f"{DEPLOY_MODEL}_threshold.joblib"

model = joblib.load(MODEL_PATH)
threshold = float(joblib.load(THRESHOLD_PATH)) if THRESHOLD_PATH.exists() else 0.5
logger.info(f"Model yüklendi: {MODEL_PATH} | Threshold: {threshold:.4f}")

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
N_FEATURES = len(FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Çeviri
# ---------------------------------------------------------------------------
TRANSLATIONS = {
    "tr": {
        "title": "Kredi Kartı Dolandırıcılık Tespiti",
        "subtitle": "Kredi kartı işlemlerini gerçek zamanlı olarak analiz eden yapay zeka sistemi.\nAşağıdan bir örnek seçin veya kendi verilerinizi girin.",
        "quick_test": "### Hızlı Test",
        "examples_note": "Örnek butonları, veri setindeki gerçek işlemlerden seçilmiştir.",
        "fraud_btn": "Yüksek Risk (Gerçek Fraud)",
        "normal_btn": "Düşük Risk (Gerçek Normal)",
        "borderline_btn": "Sınırda İşlem (Gerçek Veri)",
        "manual_input": "Manuel Veri Girişi (İleri Seviye)",
        "manual_desc": "V1–V28, PCA dönüşümü uygulanmış anonim banka özellikleridir. Bu alanları doldurmak için orijinal veri setine ihtiyaç duyulur.",
        "analyze_btn": "Analiz Et",
        "result_title": "### Sonuç",
        "footer": "**Veri Seti:** [ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284.807 işlem, %0.17 fraud\n**Model:** XGBoost | ROC-AUC: 0.975 | Karar Eşiği: F2-score ile optimize edildi",
        "fraud_prob": "Fraud Olasılığı",
        "transaction": "İşlem",
        "risk_score": "Risk Skoru",
        "action": "Aksiyon",
        "action_high": "İncele / Blokla",
        "action_mid": "Manuel İnceleme Önerilir",
        "action_low": "Onayla / İzle",
        "model_details": "Model Detayları",
        "algorithm": "Algoritma",
        "threshold_label": "Karar Eşiği",
        "raw_prob": "Ham Olasılık",
        "decision_rule": "Karar Kuralı",
        "top_factors": "En Etkili Faktörler",
        "factor_up": "fraud riskini artırıyor",
        "factor_down": "fraud riskini azaltıyor",
        "decision": "Karar",
        "pca_note": "V1–V28 özellikleri PCA ile anonimleştirilmiş banka verisidir. Threshold, F2-score (recall ağırlıklı) ile optimize edilmiştir.",
        "high_risk": "Yüksek Risk",
        "mid_risk": "Orta Risk",
        "low_risk": "Düşük Risk",
        "fraud_verdict": "DOLANDIRICILIK TESPİT EDİLDİ",
        "suspect_verdict": "ŞÜPHELİ İŞLEM",
        "normal_verdict": "NORMAL İŞLEM",
        "fraud_exp": "Bu işlem model tarafından yüksek olasılıkla dolandırıcılık olarak puanlandı.",
        "suspect_exp": "Bu işlem sınırda risk profilindedir. Manuel inceleme önerilir.",
        "normal_exp": "Bu işlem düşük risk profilindedir. Şüpheli bir sinyal tespit edilmedi.",
        "error": "Hata",
    },
    "en": {
        "title": "Credit Card Fraud Detection",
        "subtitle": "An AI system that analyzes credit card transactions in real time.\nSelect an example below or enter your own data.",
        "quick_test": "### Quick Test",
        "examples_note": "Example buttons are selected from real transactions in the dataset.",
        "fraud_btn": "High Risk (Real Fraud)",
        "normal_btn": "Low Risk (Real Normal)",
        "borderline_btn": "Borderline (Real Transaction)",
        "manual_input": "Manual Data Entry (Advanced)",
        "manual_desc": "V1–V28 are anonymized bank features transformed with PCA. You need the original dataset to fill these fields.",
        "analyze_btn": "Analyze",
        "result_title": "### Result",
        "footer": "**Dataset:** [ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 0.17% fraud\n**Model:** XGBoost | ROC-AUC: 0.975 | Decision threshold optimized with F2-score",
        "fraud_prob": "Fraud Probability",
        "transaction": "Transaction",
        "risk_score": "Risk Score",
        "action": "Action",
        "action_high": "Review / Block Transaction",
        "action_mid": "Manual Review Recommended",
        "action_low": "Approve / Monitor",
        "model_details": "Model Details",
        "algorithm": "Algorithm",
        "threshold_label": "Decision Threshold",
        "raw_prob": "Raw Probability",
        "decision_rule": "Decision Rule",
        "top_factors": "Top Contributing Factors",
        "factor_up": "increases fraud risk",
        "factor_down": "decreases fraud risk",
        "decision": "Decision",
        "pca_note": "V1–V28 are PCA-anonymized bank features. Threshold is optimized using F2-score (recall-weighted).",
        "high_risk": "High Risk",
        "mid_risk": "Medium Risk",
        "low_risk": "Low Risk",
        "fraud_verdict": "FRAUD DETECTED",
        "suspect_verdict": "SUSPICIOUS TRANSACTION",
        "normal_verdict": "NORMAL TRANSACTION",
        "fraud_exp": "This transaction is scored as highly likely fraudulent by the model.",
        "suspect_exp": "This transaction falls into a borderline risk profile. Manual review is recommended.",
        "normal_exp": "This transaction has a low risk profile. No suspicious signal was detected.",
        "error": "Error",
    },
}


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API başlatılıyor...")
    yield

api = FastAPI(title="Fraud Detection API", lifespan=lifespan)


class Transaction(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if len(v) != N_FEATURES:
            raise ValueError(f"{N_FEATURES} features required, got {len(v)}.")
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("Features must not contain NaN or infinite values.")
        return v


@api.post("/predict")
def predict_api(transaction: Transaction):
    try:
        features = np.array(transaction.features).reshape(1, -1)
        prob = float(model.predict_proba(features)[0][1])
        pred = int(prob >= threshold)
        logger.info(f"Tahmin: {'FRAUD' if pred else 'Normal'} | Olasılık: {prob:.4f}")
        return {"prediction": pred, "fraud_probability": prob, "threshold": threshold}
    except Exception as e:
        logger.error(f"Tahmin hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/health")
def health():
    return {"status": "ok", "model": DEPLOY_MODEL, "threshold": threshold}


# ---------------------------------------------------------------------------
# Örnek işlemler
# ---------------------------------------------------------------------------
FRAUD_EXAMPLE = [
    406.0, -2.3122, 1.9520, -1.6099, 3.9979, -0.5222, -1.4265, -2.5374,
    1.3917, -2.7701, -2.7723, 3.2020, -2.8999, -0.5952, -4.2893, 0.3897,
    -1.1407, -2.8301, -0.0168, 0.4170, 0.1269, 0.5172, -0.0350, -0.4652,
    0.3202, 0.0445, 0.1778, 0.2611, -0.1433, 0.0
]

NORMAL_EXAMPLE = [
    0.0, -1.3598, -0.0728, 2.5363, 1.3782, -0.3383, 0.4624, 0.2396,
    0.0987, 0.3638, 0.0908, -0.5516, -0.6178, -0.9914, -0.3112, 1.4682,
    -0.4704, 0.2080, 0.0258, 0.4040, 0.2514, -0.0183, 0.2778, -0.1105,
    0.0669, 0.1285, -0.1891, 0.1336, -0.0211, 149.62
]

BORDERLINE_EXAMPLE = [
    18709.0, -4.7353, 5.2251, -10.0603, 5.8162, -5.2049, -3.3476, -8.8116,
    3.4952, -3.2722, -10.8864, 9.4217, -14.5727, 0.1540, -12.8783, -0.4991,
    -9.7393, -16.3541, -5.0958, 1.9606, 1.3658, 1.5077, -0.3854, -0.0609,
    -0.0137, 0.0512, -0.3014, 2.0519, 0.6555, 89.99
]


# ---------------------------------------------------------------------------
# Tahmin fonksiyonu
# ---------------------------------------------------------------------------
def get_top_contributing_factors(features: list[float], top_k: int = 3):
    try:
        booster = model.get_booster()
        dmatrix = xgb.DMatrix(np.array(features).reshape(1, -1), feature_names=FEATURE_NAMES)
        contributions = booster.predict(dmatrix, pred_contribs=True)[0][:-1]
        top_idx = np.argsort(np.abs(contributions))[-top_k:][::-1]
        return [(FEATURE_NAMES[i], float(contributions[i])) for i in top_idx]
    except Exception as e:
        logger.warning(f"Contrib hesaplanamadı: {e}")
        return []


def predict_gradio(*args):
    t = TRANSLATIONS["tr"]
    features = list(args)
    try:
        transaction = Transaction(features=features)
    except Exception as e:
        error_html = f"""
        <div style="background:#fee2e2;border:1px solid #fca5a5;border-radius:12px;padding:20px;text-align:center;">
            <p style="color:#dc2626;font-size:16px;">{t['error']}: {e}</p>
        </div>"""
        return error_html, ""

    result = predict_api(transaction)
    prob = result["fraud_probability"]
    pred = result["prediction"]
    pct = prob * 100

    amount = float(features[-1])
    review_floor = max(0.50, threshold * 0.80)

    if prob >= threshold:
        risk_label, risk_color, bg_color, border_color = t["high_risk"], "#dc2626", "#fee2e2", "#fca5a5"
        verdict, explanation = t["fraud_verdict"], t["fraud_exp"]
        action_text = t["action_high"]
    elif prob >= review_floor:
        risk_label, risk_color, bg_color, border_color = t["mid_risk"], "#d97706", "#fef3c7", "#fcd34d"
        verdict, explanation = t["suspect_verdict"], t["suspect_exp"]
        action_text = t["action_mid"]
    else:
        risk_label, risk_color, bg_color, border_color = t["low_risk"], "#16a34a", "#dcfce7", "#86efac"
        verdict, explanation = t["normal_verdict"], t["normal_exp"]
        action_text = t["action_low"]

    top_factors = get_top_contributing_factors(features, top_k=3)
    top_factor_items = []
    for name, value in top_factors:
        direction = t["factor_up"] if value >= 0 else t["factor_down"]
        top_factor_items.append(
            f"<li style='display:flex;justify-content:space-between;gap:10px;padding:4px 0;'>"
            f"<span style='color:#334155;font-weight:600;'>{name}</span>"
            f"<span style='color:#0f172a;font-weight:700;'>{value:+.3f} ({direction})</span>"
            f"</li>"
        )
    factor_html = "".join(top_factor_items) if top_factor_items else "<li style='color:#64748b;'>-</li>"

    result_html = f"""
    <div style="background:{bg_color};border:2px solid {border_color};border-radius:16px;padding:28px;font-family:'Segoe UI',sans-serif;color:#111827;">
        <div style="text-align:center;margin-bottom:20px;">
            <h2 style="color:{risk_color};margin:8px 0;font-size:30px;letter-spacing:0.5px;">{verdict}</h2>
            <span style="background:{risk_color};color:white;padding:6px 14px;border-radius:999px;font-size:13px;font-weight:700;text-transform:uppercase;">{risk_label}</span>
        </div>
        <div style="margin:0 0 14px 0;padding:14px;border-radius:12px;background:rgba(255,255,255,0.58);border:1px solid rgba(15,23,42,0.10);">
            <table style="width:100%;border-collapse:collapse;">
                <tr><td style="padding:6px 0;color:#475569;font-weight:700;">{t['transaction']}</td><td style="padding:6px 0;text-align:right;color:#111827;font-weight:700;">${amount:,.2f}</td></tr>
                <tr><td style="padding:6px 0;color:#475569;font-weight:700;">{t['risk_score']}</td><td style="padding:6px 0;text-align:right;color:#111827;font-weight:700;">{prob:.6f}</td></tr>
                <tr><td style="padding:6px 0;color:#475569;font-weight:700;">{t['action']}</td><td style="padding:6px 0;text-align:right;color:{risk_color};font-weight:700;">{action_text}</td></tr>
            </table>
        </div>
        <div style="margin:20px 0;">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#374151;font-weight:600;">{t['fraud_prob']}</span>
                <span style="color:{risk_color};font-weight:bold;font-size:18px;">%{pct:.4f}</span>
            </div>
            <div style="background:#e5e7eb;border-radius:999px;height:14px;overflow:hidden;">
                <div style="background:{risk_color};width:{int(pct)}%;height:100%;border-radius:999px;"></div>
            </div>
        </div>
        <p style="color:#334155;font-size:15px;line-height:1.6;margin:16px 0 0 0;">{explanation}</p>
    </div>
    """

    detail_html = f"""
    <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;padding:16px;font-family:'Segoe UI',sans-serif;font-size:13px;color:#111827;">
        <h4 style="margin:0 0 12px 0;color:#374151;">{t['model_details']}</h4>
        <table style="width:100%;border-collapse:collapse;">
            <tr><td style="padding:4px 0;color:#4b5563;">{t['algorithm']}</td><td style="text-align:right;font-weight:600;color:#111827;">XGBoost</td></tr>
            <tr><td style="padding:4px 0;color:#4b5563;">ROC-AUC</td><td style="text-align:right;font-weight:600;color:#111827;">0.975</td></tr>
            <tr><td style="padding:4px 0;color:#4b5563;">{t['threshold_label']}</td><td style="text-align:right;font-weight:600;color:#111827;">{threshold:.4f}</td></tr>
            <tr><td style="padding:4px 0;color:#4b5563;">{t['raw_prob']}</td><td style="text-align:right;font-weight:600;color:#111827;">{prob:.6f}</td></tr>
            <tr><td style="padding:4px 0;color:#4b5563;">{t['decision_rule']}</td><td style="text-align:right;font-weight:600;color:#111827;">score {'>=' if pred else '<'} {threshold:.4f}</td></tr>
            <tr><td style="padding:4px 0;color:#6b7280;">{t['decision']}</td><td style="text-align:right;font-weight:600;color:{risk_color};">{'Fraud' if pred else 'Normal'}</td></tr>
        </table>
        <h4 style="margin:14px 0 8px 0;color:#374151;">{t['top_factors']}</h4>
        <ul style="list-style:none;padding:0;margin:0;">
            {factor_html}
        </ul>
        <p style="margin:12px 0 0 0;color:#6b7280;font-size:11px;">{t['pca_note']}</p>
    </div>
    """

    return result_html, detail_html


def update_ui(lang):
    t = TRANSLATIONS[lang]
    return (
        t["title"],
        t["subtitle"],
        t["quick_test"],
        t["fraud_btn"],
        t["normal_btn"],
        t["analyze_btn"],
        t["result_title"],
        t["manual_input"],
        t["footer"],
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
APP_CSS = """
.gradio-container {
  max-width: 1240px !important;
  margin: 0 auto !important;
}
#demo-note p {
  color: #94a3b8 !important;
  margin-top: -4px !important;
}
#btn-predict button {
  font-weight: 700 !important;
  letter-spacing: 0.2px !important;
}
"""

with gr.Blocks(title="Credit Card Fraud Detection", css=APP_CSS) as demo:

    # Header
    header_title = gr.Markdown("# Kredi Kartı Dolandırıcılık Tespiti")
    header_sub = gr.Markdown("Kredi kartı işlemlerini gerçek zamanlı olarak analiz eden yapay zeka sistemi.\nAşağıdan bir örnek seçin veya kendi verilerinizi girin.")

    # Hızlı test
    quick_test_md = gr.Markdown("### Hızlı Test")
    examples_note_md = gr.Markdown(
        "Örnek butonları veri setindeki gerçek işlemlerden seçilmiştir.",
        elem_id="demo-note",
    )
    with gr.Row():
        btn_fraud = gr.Button("Yüksek Risk (Gerçek Fraud)", variant="secondary")
        btn_normal = gr.Button("Düşük Risk (Gerçek Normal)", variant="secondary")
        btn_borderline = gr.Button("Sınırda İşlem (Gerçek Veri)", variant="secondary")

    # Manuel giriş
    with gr.Accordion("Manuel Veri Girişi (İleri Seviye)", open=False) as manual_accordion:
        gr.Markdown("V1–V28, PCA dönüşümü uygulanmış anonim banka özellikleridir. Bu alanları doldurmak için orijinal veri setine ihtiyaç duyulur.")
        inputs = [gr.Number(label=name, value=0.0) for name in FEATURE_NAMES]

    # Analiz butonu
    btn_predict = gr.Button("Analiz Et", variant="primary", size="lg", elem_id="btn-predict")

    # Sonuç
    result_title_md = gr.Markdown("### Sonuç")
    with gr.Row():
        with gr.Column(scale=3):
            out_result = gr.HTML()
        with gr.Column(scale=2):
            out_detail = gr.HTML()

    # Footer
    footer_md = gr.Markdown("**Veri Seti:** [ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284.807 işlem, %0.17 fraud\n**Model:** XGBoost | ROC-AUC: 0.975 | Threshold: F2-score ile optimize edildi")

    # Tahmin
    btn_predict.click(
        fn=predict_gradio,
        inputs=inputs,
        outputs=[out_result, out_detail],
    )
    btn_fraud.click(lambda: FRAUD_EXAMPLE, outputs=inputs)
    btn_normal.click(lambda: NORMAL_EXAMPLE, outputs=inputs)
    btn_borderline.click(lambda: BORDERLINE_EXAMPLE, outputs=inputs)

# ---------------------------------------------------------------------------
# Mount & run
# ---------------------------------------------------------------------------
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
