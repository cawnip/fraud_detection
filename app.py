import logging
import numpy as np
import joblib
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
        "fraud_btn": "🚨  Sahte İşlem Örneği",
        "normal_btn": "✅  Normal İşlem Örneği",
        "borderline_btn": "⚖️  Sınırda Risk Örneği",
        "edge_btn": "💸  Uç Tutar Örneği",
        "manual_input": "Manuel Veri Girişi (İleri Seviye)",
        "manual_desc": "V1–V28, PCA dönüşümü uygulanmış anonim banka özellikleridir. Bu alanları doldurmak için orijinal veri setine ihtiyaç duyulur.",
        "analyze_btn": "Analiz Et",
        "result_title": "### Sonuç",
        "settings_title": "### Ayarlar",
        "language_label": "Dil / Language",
        "footer": "**Veri Seti:** [ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284.807 işlem, %0.17 fraud\n**Model:** XGBoost | ROC-AUC: 0.975 | Threshold: F2-score ile optimize edildi",
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
        "decision": "Karar",
        "pca_note": "V1–V28 özellikleri PCA ile anonimleştirilmiş banka verisidir. Threshold, F2-score (recall ağırlıklı) ile optimize edilmiştir.",
        "high_risk": "Yüksek Risk",
        "mid_risk": "Orta Risk",
        "low_risk": "Düşük Risk",
        "fraud_verdict": "DOLANDIRICILIK TESPİT EDİLDİ",
        "suspect_verdict": "ŞÜPHELİ İŞLEM",
        "normal_verdict": "NORMAL İŞLEM",
        "fraud_exp": "Bu işlem, modelimiz tarafından <b>yüksek olasılıkla dolandırıcılık</b> olarak sınıflandırıldı. İşlem incelemeye alınmalıdır.",
        "suspect_exp": "Bu işlem <b>şüpheli</b> görünmektedir. Manuel inceleme önerilir.",
        "normal_exp": "Bu işlem <b>normal</b> görünmektedir. Herhangi bir şüpheli durum tespit edilmedi.",
        "error": "Hata",
    },
    "en": {
        "title": "Credit Card Fraud Detection",
        "subtitle": "An AI system that analyzes credit card transactions in real time.\nSelect an example below or enter your own data.",
        "quick_test": "### Quick Test",
        "fraud_btn": "🚨  Fraudulent Transaction Example",
        "normal_btn": "✅  Normal Transaction Example",
        "borderline_btn": "⚖️  Borderline Risk Example",
        "edge_btn": "💸  Extreme Amount Example",
        "manual_input": "Manual Data Entry (Advanced)",
        "manual_desc": "V1–V28 are anonymized bank features transformed with PCA. You need the original dataset to fill these fields.",
        "analyze_btn": "Analyze",
        "result_title": "### Result",
        "settings_title": "### Settings",
        "language_label": "Dil / Language",
        "footer": "**Dataset:** [ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 0.17% fraud\n**Model:** XGBoost | ROC-AUC: 0.975 | Threshold optimized with F2-score",
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
        "decision": "Decision",
        "pca_note": "V1–V28 are PCA-anonymized bank features. Threshold is optimized using F2-score (recall-weighted).",
        "high_risk": "High Risk",
        "mid_risk": "Medium Risk",
        "low_risk": "Low Risk",
        "fraud_verdict": "FRAUD DETECTED",
        "suspect_verdict": "SUSPICIOUS TRANSACTION",
        "normal_verdict": "NORMAL TRANSACTION",
        "fraud_exp": "This transaction has been classified as <b>highly likely fraudulent</b> by our model. The transaction should be reviewed.",
        "suspect_exp": "This transaction appears <b>suspicious</b>. Manual review is recommended.",
        "normal_exp": "This transaction appears <b>normal</b>. No suspicious activity detected.",
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
    212.3380, -1.8579, 0.9862, 0.3678, 2.7483, -0.4345, -0.5255, -1.2128,
    0.7749, -1.2752, -1.4066, 1.4115, -1.8113, -0.7842, -2.3917, 0.9041,
    -0.8210, -1.3809, 0.0035, 0.4108, 0.1863, 0.2618, 0.1142, -0.2960,
    0.1994, 0.0846, 0.0028, 0.2003, -0.0850, 71.3687
]

EDGE_AMOUNT_EXAMPLE = [
    10000.0, -1.3598, -0.0728, 2.5363, 1.3782, -0.3383, 0.4624, 0.2396,
    0.0987, 0.3638, 0.0908, -0.5516, -0.6178, -0.9914, -0.3112, 1.4682,
    -0.4704, 0.2080, 0.0258, 0.4040, 0.2514, -0.0183, 0.2778, -0.1105,
    0.0669, 0.1285, -0.1891, 0.1336, -0.0211, 5000.0
]


# ---------------------------------------------------------------------------
# Tahmin fonksiyonu
# ---------------------------------------------------------------------------
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

    if pct >= 80:
        risk_label, risk_color, bg_color, border_color = t["high_risk"], "#dc2626", "#fee2e2", "#fca5a5"
        icon, verdict, explanation = "🚨", t["fraud_verdict"], t["fraud_exp"]
        action_text = t["action_high"]
    elif pct >= 40:
        risk_label, risk_color, bg_color, border_color = t["mid_risk"], "#d97706", "#fef3c7", "#fcd34d"
        icon, verdict, explanation = "⚠️", t["suspect_verdict"], t["suspect_exp"]
        action_text = t["action_mid"]
    else:
        risk_label, risk_color, bg_color, border_color = t["low_risk"], "#16a34a", "#dcfce7", "#86efac"
        icon, verdict, explanation = "✅", t["normal_verdict"], t["normal_exp"]
        action_text = t["action_low"]

    # Dark/light tema farklarında metinlerin kaybolmaması için sabit renk kullan
    explanation_safe = explanation.replace("<b>", "<b style='color:#111827;font-weight:700;'>")

    result_html = f"""
    <div style="background:{bg_color};border:2px solid {border_color};border-radius:16px;padding:28px;font-family:sans-serif;color:#111827;">
        <div style="text-align:center;margin-bottom:20px;">
            <span style="font-size:48px;">{icon}</span>
            <h2 style="color:{risk_color};margin:8px 0;font-size:24px;letter-spacing:1px;">{verdict}</h2>
            <span style="background:{risk_color};color:white;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:bold;">{risk_label}</span>
        </div>
        <div style="margin:0 0 14px 0;padding:14px;border-radius:12px;background:rgba(255,255,255,0.55);border:1px solid rgba(0,0,0,0.07);">
            <p style="margin:0 0 6px 0;color:#1f2937;font-size:15px;"><span style="font-weight:700;color:#374151;">💳 {t['transaction']}:</span> <span style="font-weight:700;color:#1f2937;">${amount:,.2f}</span></p>
            <p style="margin:0 0 6px 0;color:#1f2937;font-size:15px;"><span style="font-weight:700;color:#374151;">📈 {t['risk_score']}:</span> <span style="font-weight:700;color:#1f2937;">{prob:.4f}</span></p>
            <p style="margin:0;color:#111827;font-size:15px;"><span style="font-weight:700;color:#374151;">{t['action']}:</span> <span style="color:{risk_color};font-weight:700;">{action_text}</span></p>
        </div>
        <div style="margin:20px 0;">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="color:#374151;font-weight:600;">{t['fraud_prob']}</span>
                <span style="color:{risk_color};font-weight:bold;font-size:18px;">%{pct:.2f}</span>
            </div>
            <div style="background:#e5e7eb;border-radius:999px;height:14px;overflow:hidden;">
                <div style="background:{risk_color};width:{int(pct)}%;height:100%;border-radius:999px;"></div>
            </div>
        </div>
        <p style="color:#374151;font-size:14px;line-height:1.6;margin:16px 0 0 0;">{explanation_safe}</p>
    </div>
    """

    detail_html = f"""
    <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;padding:16px;font-family:sans-serif;font-size:13px;color:#111827;">
        <h4 style="margin:0 0 12px 0;color:#374151;">{t['model_details']}</h4>
        <table style="width:100%;border-collapse:collapse;">
            <tr><td style="padding:4px 0;color:#4b5563;">{t['algorithm']}</td><td style="text-align:right;font-weight:600;color:#111827;">XGBoost</td></tr>
            <tr><td style="padding:4px 0;color:#4b5563;">ROC-AUC</td><td style="text-align:right;font-weight:600;color:#111827;">0.975</td></tr>
            <tr><td style="padding:4px 0;color:#4b5563;">{t['threshold_label']}</td><td style="text-align:right;font-weight:600;color:#111827;">{threshold:.4f}</td></tr>
            <tr><td style="padding:4px 0;color:#4b5563;">{t['raw_prob']}</td><td style="text-align:right;font-weight:600;color:#111827;">{prob:.6f}</td></tr>
            <tr><td style="padding:4px 0;color:#6b7280;">{t['decision']}</td><td style="text-align:right;font-weight:600;color:{risk_color};">{'Fraud' if pred else 'Normal'}</td></tr>
        </table>
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
with gr.Blocks(title="Credit Card Fraud Detection") as demo:

    # Header
    header_title = gr.Markdown("# Kredi Kartı Dolandırıcılık Tespiti")
    header_sub = gr.Markdown("Kredi kartı işlemlerini gerçek zamanlı olarak analiz eden yapay zeka sistemi.\nAşağıdan bir örnek seçin veya kendi verilerinizi girin.")

    # Hızlı test
    quick_test_md = gr.Markdown("### Hızlı Test")
    with gr.Row():
        btn_fraud = gr.Button("🚨  Sahte İşlem Örneği", variant="secondary")
        btn_normal = gr.Button("✅  Normal İşlem Örneği", variant="secondary")
    with gr.Row():
        btn_borderline = gr.Button("⚖️  Sınırda Risk Örneği", variant="secondary")
        btn_edge = gr.Button("💸  Uç Tutar Örneği", variant="secondary")

    # Manuel giriş
    with gr.Accordion("Manuel Veri Girişi (İleri Seviye)", open=False) as manual_accordion:
        gr.Markdown("V1–V28, PCA dönüşümü uygulanmış anonim banka özellikleridir. Bu alanları doldurmak için orijinal veri setine ihtiyaç duyulur.")
        inputs = [gr.Number(label=name, value=0.0) for name in FEATURE_NAMES]

    # Analiz butonu
    btn_predict = gr.Button("Analiz Et", variant="primary", size="lg")

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
    btn_edge.click(lambda: EDGE_AMOUNT_EXAMPLE, outputs=inputs)

# ---------------------------------------------------------------------------
# Mount & run
# ---------------------------------------------------------------------------
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
