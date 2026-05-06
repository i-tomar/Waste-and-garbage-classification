import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "model/best.pth"
CLASS_NAMES = [
    "battery", "biological", "brown-glass", "cardboard",
    "clothes", "green-glass", "metal", "paper",
    "plastic", "shoes", "trash", "white-glass",
]

DISPOSAL_GUIDE = {
    "battery":     "⚠️ Hazardous Waste — Special Disposal Required",
    "biological":  "🌱 Compostable — Organic Waste Bin",
    "brown-glass": "♻️ Recyclable — Glass Recycling Bin",
    "cardboard":   "♻️ Recyclable — Paper & Cardboard Bin",
    "clothes":     "🗑️ Landfill — General Waste Bin",
    "green-glass": "♻️ Recyclable — Glass Recycling Bin",
    "metal":       "♻️ Recyclable — Metal Recycling Bin",
    "paper":       "♻️ Recyclable — Paper & Cardboard Bin",
    "plastic":     "♻️ Recyclable — Plastic Recycling Bin",
    "shoes":       "🗑️ Landfill — General Waste Bin",
    "trash":       "🗑️ Landfill — General Waste Bin",
    "white-glass": "♻️ Recyclable — Glass Recycling Bin",
}

# (text_color, bg_rgba, border_rgba)
DISPOSAL_COLORS = {
    "♻️ Recyclable — Glass Recycling Bin":      ("#10B981", "rgba(16,185,129,0.12)", "rgba(16,185,129,0.35)"),
    "♻️ Recyclable — Paper & Cardboard Bin":    ("#10B981", "rgba(16,185,129,0.12)", "rgba(16,185,129,0.35)"),
    "♻️ Recyclable — Metal Recycling Bin":      ("#10B981", "rgba(16,185,129,0.12)", "rgba(16,185,129,0.35)"),
    "♻️ Recyclable — Plastic Recycling Bin":    ("#10B981", "rgba(16,185,129,0.12)", "rgba(16,185,129,0.35)"),
    "🌱 Compostable — Organic Waste Bin":       ("#22D3EE", "rgba(34,211,238,0.12)", "rgba(34,211,238,0.35)"),
    "⚠️ Hazardous Waste — Special Disposal Required": ("#F59E0B", "rgba(245,158,11,0.12)", "rgba(245,158,11,0.35)"),
    "🗑️ Landfill — General Waste Bin":         ("#94A3B8", "rgba(148,163,184,0.12)", "rgba(148,163,184,0.35)"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS – Glassmorphism / Deep Navy Design System
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ════════════════════════════════════════════
   0. Font Import
════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ════════════════════════════════════════════
   1. Design Tokens
════════════════════════════════════════════ */
:root {
  --navy-900:     #050d1a;
  --navy-800:     #0a1628;
  --navy-700:     #0f1f38;
  --navy-600:     #162840;
  --navy-500:     #1e3252;
  --card-bg:      rgba(15, 31, 56, 0.7);
  --card-border:  rgba(255,255,255,0.08);
  --glass-blur:   18px;
  --blue:         #3B82F6;
  --blue-glow:    rgba(59,130,246,0.18);
  --green:        #10B981;
  --green-glow:   rgba(16,185,129,0.15);
  --text-1:       #e2eaf6;
  --text-2:       #94a3b8;
  --text-3:       #64748b;
  --radius-sm:    8px;
  --radius-md:    12px;
  --radius-lg:    18px;
  --radius-xl:    26px;
  --transition:   0.3s cubic-bezier(0.4,0,0.2,1);
  /* ── CENTERED LAYOUT ── */
  --max-w:        860px;
}

/* ════════════════════════════════════════════
   2. Global Reset
════════════════════════════════════════════ */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp {
  background: radial-gradient(ellipse at 20% 10%, rgba(59,130,246,0.07) 0%, transparent 55%),
              radial-gradient(ellipse at 80% 80%, rgba(16,185,129,0.05) 0%, transparent 50%),
              var(--navy-900) !important;
  min-height: 100vh;
}
#MainMenu, footer, header, .stDeployButton { display: none !important; }

/* ════════════════════════════════════════════
   3. CENTERING WRAPPER
   Forces the Streamlit block container to stay
   constrained and centered — fixes full-width stretch
════════════════════════════════════════════ */
.block-container {
  max-width: var(--max-w) !important;
  padding-left:  1.5rem !important;
  padding-right: 1.5rem !important;
  padding-top:   0 !important;
  margin: 0 auto !important;
}

/* ════════════════════════════════════════════
   4. Header Banner (full bleed behind, text centered)
════════════════════════════════════════════ */
.app-header {
  text-align: center;
  padding: 3rem 1.5rem 2.5rem;
  background: linear-gradient(160deg,
    rgba(59,130,246,0.12) 0%,
    rgba(16,185,129,0.06) 60%,
    transparent 100%);
  border-bottom: 1px solid var(--card-border);
  margin: -1rem calc(-50vw + 50%) 2.5rem;   /* spans viewport edge-to-edge */
  position: relative;
}
.app-header::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: var(--navy-900);
  z-index: -1;
}
.app-header-inner {
  max-width: var(--max-w);
  margin: 0 auto;
}
.app-header h1 {
  font-size: clamp(1.8rem, 4vw, 2.6rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #adc6ff 10%, #3B82F6 50%, #10B981 90%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 0.5rem;
  line-height: 1.1;
}
.app-header p {
  color: var(--text-2);
  font-size: 0.95rem;
  font-weight: 400;
  letter-spacing: 0.02em;
  margin: 0;
}
.header-pills {
  display: flex;
  gap: 0.5rem;
  justify-content: center;
  margin-top: 1.2rem;
  flex-wrap: wrap;
}
.header-pill {
  padding: 0.3rem 0.9rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border: 1px solid;
}
.pill-blue  { color: var(--blue);  background: rgba(59,130,246,0.1);  border-color: rgba(59,130,246,0.3); }
.pill-green { color: var(--green); background: rgba(16,185,129,0.1);  border-color: rgba(16,185,129,0.3); }
.pill-gray  { color: var(--text-2); background: rgba(148,163,184,0.1); border-color: rgba(148,163,184,0.25); }

/* ════════════════════════════════════════════
   5. Glass Card Component
════════════════════════════════════════════ */
.glass-card {
  background: var(--card-bg);
  backdrop-filter: blur(var(--glass-blur));
  -webkit-backdrop-filter: blur(var(--glass-blur));
  border: 1px solid var(--card-border);
  border-radius: var(--radius-lg);
  padding: 1.75rem 2rem;
  margin-bottom: 1.25rem;
  box-shadow:
    0 1px 0 0 rgba(255,255,255,0.04) inset,
    0 12px 40px rgba(0,0,0,0.35),
    0 4px 16px rgba(0,0,0,0.2);
  transition: border-color var(--transition), box-shadow var(--transition);
}
.glass-card:hover {
  border-color: rgba(59,130,246,0.2);
  box-shadow:
    0 1px 0 0 rgba(255,255,255,0.04) inset,
    0 12px 40px rgba(0,0,0,0.4),
    0 0 0 1px rgba(59,130,246,0.08);
}
.card-label {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-3);
  margin-bottom: 1rem;
}

/* ════════════════════════════════════════════
   6. Upload Zone
════════════════════════════════════════════ */
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploader"] section {
  background: rgba(10,22,40,0.6) !important;
  border: 2px dashed rgba(59,130,246,0.28) !important;
  border-radius: var(--radius-xl) !important;
  padding: 3.5rem 2rem !important;
  transition: all var(--transition) !important;
  text-align: center;
}
[data-testid="stFileUploader"] section:hover {
  border-color: var(--blue) !important;
  background: rgba(59,130,246,0.04) !important;
  box-shadow:
    inset 0 0 40px rgba(59,130,246,0.06),
    0 0 24px var(--blue-glow) !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span { color: var(--text-2) !important; }
[data-testid="stFileUploader"] button {
  background: rgba(59,130,246,0.12) !important;
  border: 1px solid rgba(59,130,246,0.35) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--blue) !important;
  font-weight: 600 !important;
  transition: all var(--transition) !important;
}
[data-testid="stFileUploader"] button:hover {
  background: rgba(59,130,246,0.22) !important;
}

/* ════════════════════════════════════════════
   7. Image Display
════════════════════════════════════════════ */
[data-testid="stImage"] img {
  border-radius: var(--radius-md) !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.45) !important;
}

/* ════════════════════════════════════════════
   8. st.metric Override
════════════════════════════════════════════ */
[data-testid="stMetric"] {
  background: rgba(10,22,40,0.5) !important;
  border: 1px solid var(--card-border) !important;
  border-radius: var(--radius-md) !important;
  padding: 1.1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] > div {
  font-size: 0.7rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: var(--text-3) !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.6rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em !important;
  color: #adc6ff !important;
}

/* ════════════════════════════════════════════
   9. st.progress Override
════════════════════════════════════════════ */
[data-testid="stProgressBar"] > div {
  background: rgba(59,130,246,0.15) !important;
  border-radius: 999px !important;
  height: 10px !important;
}
[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, var(--blue), var(--green)) !important;
  border-radius: 999px !important;
}

/* ════════════════════════════════════════════
   10. Bar Chart
════════════════════════════════════════════ */
[data-testid="stVegaLiteChart"] {
  background: transparent !important;
  border: none !important;
}

/* ════════════════════════════════════════════
   11. Alerts
════════════════════════════════════════════ */
[data-testid="stAlert"] {
  border-radius: var(--radius-md) !important;
  border-width: 1px !important;
}

/* ════════════════════════════════════════════
   12. Spinner
════════════════════════════════════════════ */
.stSpinner > div { border-top-color: var(--blue) !important; }

/* ════════════════════════════════════════════
   13. Divider
════════════════════════════════════════════ */
hr { border-color: var(--card-border) !important; margin: 1.5rem 0 !important; }

/* ════════════════════════════════════════════
   14. Scrollbar
════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; background: var(--navy-900); }
::-webkit-scrollbar-thumb { background: var(--navy-600); border-radius: 999px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading & Caching  (backend logic — unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Loading model weights…")
def load_model():
    """Loads the PyTorch ResNet18 model and caches it."""
    m = models.resnet18(pretrained=False)
    m.fc = torch.nn.Linear(m.fc.in_features, len(CLASS_NAMES))
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Image Preprocessing  (backend logic — unchanged)
# ─────────────────────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])


def predict_image(pil_image, model):
    """Runs inference. Returns (class_name, confidence_float, prob_dataframe)."""
    img = pil_image.convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits[0], dim=0)
        conf, idx = torch.max(probs, 0)
    prob_pct = (probs * 100).cpu().numpy()
    prob_df = pd.DataFrame({
        "Category": [n.replace("-", " ").title() for n in CLASS_NAMES],
        "Confidence (%)": prob_pct,
    }).set_index("Category")
    return CLASS_NAMES[idx.item()], conf.item(), prob_df


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    model = load_model()

    # ── 1. Header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
      <div class="app-header-inner">
        <h1>♻️ Smart Waste Classifier</h1>
        <p>AI-powered recyclable &amp; waste sorting — powered by ResNet-18 &amp; PyTorch</p>
        <div class="header-pills">
          <span class="header-pill pill-blue">ResNet-18</span>
          <span class="header-pill pill-green">88% Accuracy</span>
          <span class="header-pill pill-gray">12 Categories</span>
          <span class="header-pill pill-gray">PyTorch 2.7</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. Hero Upload Section ──────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="card-label">📤 Upload Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your waste image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        help="Accepted formats: JPG, JPEG, PNG, WEBP",
    )
    st.markdown(
        '<p style="text-align:center;color:#64748b;font-size:0.8rem;margin-top:0.5rem;">'
        '📌 Supports JPG · JPEG · PNG · WebP</p>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 3. Results (conditionally rendered) ────────────────────────────────
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)

            # Validate it is a proper image
            pil_image.verify()
            pil_image = Image.open(uploaded_file)   # re-open after verify

            with st.spinner("🔬  Analyzing image — running inference…"):
                pred_class, confidence, prob_df = predict_image(pil_image, model)

            display_name = pred_class.replace("-", " ").title()
            action       = DISPOSAL_GUIDE.get(pred_class, "Unknown")
            txt_col, bg_col, bd_col = DISPOSAL_COLORS.get(
                action, ("#94A3B8", "rgba(148,163,184,0.12)", "rgba(148,163,184,0.35)")
            )
            conf_pct = confidence * 100

            # ── 3a. Two-column result card ──
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<p class="card-label">🧪 Analysis Results</p>', unsafe_allow_html=True)

            col_img, col_info = st.columns([1, 1], gap="large")

            with col_img:
                st.image(pil_image, caption="Uploaded Image", use_container_width=True)

            with col_info:
                st.metric(label="Predicted Class", value=display_name)

                st.markdown(f"""
                <div style="margin: 1.1rem 0 0.4rem;">
                  <p style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                             letter-spacing:0.1em;color:#64748b;margin-bottom:0.55rem;">
                    Confidence Score
                  </p>
                  <p style="font-size:2rem;font-weight:800;letter-spacing:-0.03em;
                             color:{txt_col};margin:0 0 0.6rem;">{conf_pct:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(min(confidence, 1.0))

                st.markdown(f"""
                <div style="margin-top:1.4rem;">
                  <p style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                             letter-spacing:0.1em;color:#64748b;margin-bottom:0.6rem;">
                    Disposal Guide
                  </p>
                  <span style="
                    display:inline-block;
                    padding:0.55rem 1.3rem;
                    border-radius:999px;
                    font-size:0.88rem;
                    font-weight:600;
                    letter-spacing:0.01em;
                    color:{txt_col};
                    background:{bg_col};
                    border:1px solid {bd_col};">
                    {action}
                  </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # ── 3b. Streamlit alert banner ──
            if "Recyclable" in action:
                st.info(f"**Disposal Guide:** {action}", icon="♻️")
            elif "Hazardous" in action:
                st.error(f"**Disposal Guide:** {action}", icon="⚠️")
            elif "Compostable" in action:
                st.success(f"**Disposal Guide:** {action}", icon="🌱")
            else:
                st.warning(f"**Disposal Guide:** {action}", icon="🗑️")

            # ── 3c. Confidence Breakdown Chart ──
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                '<p class="card-label">📊 Full Model Confidence Breakdown</p>',
                unsafe_allow_html=True,
            )
            st.bar_chart(prob_df, color="#3B82F6")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.error(
                "**Unable to process this image.**\n\n"
                "Please make sure you uploaded a valid image file (JPG, PNG, or WebP) "
                "and that it is not corrupted.\n\n"
                f"_Technical detail: {e}_",
                icon="🚨",
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # ── 4. Footer ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
      text-align: center;
      padding: 2.5rem 0 1.5rem;
      margin-top: 3rem;
      border-top: 1px solid rgba(255,255,255,0.05);
    ">
      <p style="color:#475569;font-size:0.78rem;font-weight:500;letter-spacing:0.03em;margin:0;">
        Developed by <span style="color:#7C93D0;font-weight:600;">Ishan Singh Tomar</span>
        &nbsp;|&nbsp; CSIT 3rd Year
        &nbsp;&nbsp;·&nbsp;&nbsp;
        Powered by <span style="color:#3B82F6;">PyTorch</span>
        &nbsp;&amp;&nbsp; <span style="color:#10B981;">Streamlit</span>
      </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
