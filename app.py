"""
Smart Crop & Land Improvement System (single-file Streamlit app)
Features:
 - 36-district dropdown for Maharashtra
 - Live weather fetch (OpenWeatherMap)
 - ML-based crop suggestion (RandomForest) using small sample dataset (trainable)
 - If user picks a desired crop, compares farm inputs vs ideal crop ranges and gives suggestions
 - CSV download of result
 - PDF report download containing charts (Input vs Ideal bar chart, suitability pie chart)
Run:
    pip install streamlit requests pandas numpy scikit-learn matplotlib reportlab pillow joblib
    streamlit run app.py
Notes:
 - This is your original app code with a minimal bilingual (English / Marathi) layer added.
 - Layout and logic are kept unchanged; only textual outputs and suggestion sentences are passed
   through a translation helper so when Marathi is selected relevant outputs become Marathi.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --------------------------- CONFIG ---------------------------
st.set_page_config(page_title="Smart Crop & Land Improvement", page_icon="🌾", layout="wide")

# Put your OpenWeatherMap API key here:
OPENWEATHER_API_KEY = "217e7333d781ce97c2904a72e1e0db0e"

MODEL_FILE = "crop_model.joblib"

# --------------------------- DISTRICTS (Maharashtra - 36) ---------------------------
DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara",
    "Buldhana", "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Hingoli",
    "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai City", "Mumbai Suburban",
    "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad", "Palghar",
    "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"
]

# Marathi names mapping for districts (so UI can show Marathi when language selected)
DISTRICTS_MR_MAP = {
    "Ahmednagar": "अहमदनगर", "Akola": "अकोला", "Amravati": "अमरावती", "Aurangabad": "औरंगाबाद",
    "Beed": "बीड", "Bhandara": "भंडारा", "Buldhana": "बुलढाणा", "Chandrapur": "चंद्रपूर",
    "Dhule": "धुळे", "Gadchiroli": "गडचिरोली", "Gondia": "गोंदिया", "Hingoli": "हिंगोली",
    "Jalgaon": "जळगाव", "Jalna": "जालना", "Kolhapur": "कोल्हापूर", "Latur": "लातूर",
    "Mumbai City": "मुंबई सिटी", "Mumbai Suburban": "मुंबई उपनगरीय", "Nagpur": "नागपूर",
    "Nanded": "नांदेड", "Nandurbar": "नंदुरबार", "Nashik": "नाशिक", "Osmanabad": "उस्मानाबाद",
    "Palghar": "पालघर", "Parbhani": "परभणी", "Pune": "पुणे", "Raigad": "रायगड",
    "Ratnagiri": "रत्नागिरी", "Sangli": "सांगली", "Satara": "सातारा", "Sindhudurg": "सिंधुदुर्ग",
    "Solapur": "सोलापुर", "Thane": "ठाणे", "Wardha": "वर्धा", "Washim": "वाशीम", "Yavatmal": "यवतमाळ"
}

# Crop translations (EN -> MR)
CROP_TRANSLATIONS = {
    "rice":"तांदूळ","paddy":"तांदूळ","maize":"मका","wheat":"गहू","cotton":"कापूस",
    "chickpea":"हरभरा","millet":"बाजरी","sorghum":"ज्वारी","groundnut":"शेंगदाणा","peanut":"शेंगदाणा",
    "sugarcane":"ऊस","soybean":"सोयाबीन","sunflower":"सूर्यफूल","potato":"बटाटा","tomato":"टोमॅटो","onion":"कांदा"
}

# --------------------------- TRANSLATION HELPERS ---------------------------
# Simple translation helper: returns Marathi when selected, else English.
if "lang" not in st.session_state:
    st.session_state["lang"] = "English"  # default

def get_lang():
    return st.session_state.get("lang", "English")

def t(en_text, mr_text):
    """Return mr_text when Marathi selected, else en_text."""
    return mr_text if get_lang() == "मराठी" else en_text

def translate_crop_name(name):
    if not name:
        return ""
    key = str(name).lower()
    if get_lang() == "मराठी":
        # return Marathi translation if exists, else original (but we prefer Marathi map)
        return CROP_TRANSLATIONS.get(key, name)
    else:
        return name.upper()

# A few reusable translated fragments used in suggestions
TRANSLATED_FRAGMENTS = {
    "Increase": {"mr": "वाढवा"},
    "from": {"mr": "पासून"},
    "at least": {"mr": "किमान"},
    "apply recommended fertilizer containing N": {"mr": "शिफारस केलेले N असलेले खत वापरा"},
    "Humidity": {"mr": "आर्द्रता"},
    "is high — ensure good drainage and disease management.": {"mr": "जास्त आहे — योग्य निचरा आणि रोग नियंत्रण करा."},
    "is low — consider irrigation planning or select drought-tolerant varieties.": {"mr": "कमी आहे — सिंचन नियोजन करा किंवा दुष्काळ-प्रतिरोधक वाण निवडा."},
    "Suitable": {"mr": "योग्य"},
    "Remaining": {"mr": "बाकी"}
}

def tf(key):
    """Fetch translated fragment; falls back to key if not found."""
    if get_lang() == "मराठी":
        return TRANSLATED_FRAGMENTS.get(key, {}).get("mr", key)
    else:
        return key

# --------------------------- SAMPLE DATASET & IDEAL RANGES ---------------------------
def get_sample_dataset():
    data = [
        # N, P, K, temp, humidity, ph, rainfall, crop
        [90, 42, 43, 20.8, 82, 6.5, 200, "rice"],
        [45, 56, 10, 25.6, 85, 7.2, 100, "maize"],
        [60, 55, 44, 26.3, 80, 6.8, 120, "maize"],
        [80, 30, 30, 28.0, 70, 7.0, 20, "cotton"],
        [28, 45, 33, 22.0, 75, 6.2, 50, "chickpea"],
        [50, 75, 70, 27.0, 90, 5.6, 300, "paddy"],
        [70, 20, 30, 30.0, 60, 7.5, 10, "cotton"],
        [40, 40, 40, 21.0, 85, 6.5, 180, "rice"],
        [30, 60, 40, 23.0, 90, 6.7, 220, "rice"],
        [20, 20, 5, 24.0, 65, 6.0, 5, "millet"],
        [55, 45, 30, 24.5, 70, 6.3, 30, "sorghum"],
        [35, 70, 55, 22.5, 88, 5.8, 260, "paddy"],
        [10, 10, 10, 35.0, 40, 7.8, 2, "groundnut"],
        [65, 20, 20, 29.0, 55, 7.2, 7, "peanut"],
        [48, 60, 40, 19.0, 95, 6.0, 350, "paddy"],
        [80, 40, 60, 18.0, 90, 5.9, 400, "rice"]
    ]
    cols = ["N","P","K","temperature","humidity","ph","rainfall","crop"]
    return pd.DataFrame(data, columns=cols)

IDEAL_RANGES = {
    "rice":      {"N": (70,120), "P": (30,80), "K": (40,80), "ph": (5.5,7.0), "temperature": (18,30), "humidity": (70,95),  "rainfall": (150,500)},
    "maize":     {"N": (50,100), "P": (30,70), "K": (30,70), "ph": (5.5,7.5), "temperature": (20,32), "humidity": (50,80),  "rainfall": (50,300)},
    "wheat":     {"N": (40,90),  "P": (30,60), "K": (30,60), "ph": (6.0,7.5), "temperature": (12,25), "humidity": (40,75),  "rainfall": (30,150)},
    "cotton":    {"N": (60,120), "P": (20,50), "K": (30,80), "ph": (6.0,8.0), "temperature": (20,35), "humidity": (30,70),  "rainfall": (20,100)},
    "paddy":     {"N": (70,120), "P": (35,80), "K": (40,80), "ph": (5.0,6.5), "temperature": (20,32), "humidity": (70,95),  "rainfall": (200,500)},
    "millet":    {"N": (10,40),  "P": (10,40), "K": (10,40), "ph": (5.5,8.0), "temperature": (20,35), "humidity": (30,70),  "rainfall": (10,200)},
    "groundnut": {"N": (10,50),  "P": (20,60), "K": (20,60), "ph": (5.5,7.0), "temperature": (25,35), "humidity": (50,80),  "rainfall": (50,300)}
}

# --------------------------- ML: train/load ---------------------------
def train_and_save_model(df):
    X = df[["N","P","K","temperature","humidity","ph","rainfall"]]
    y = df["crop"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    joblib.dump(model, MODEL_FILE)
    return model, acc

def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            return model, None
        except Exception:
            pass
    df = get_sample_dataset()
    model, acc = train_and_save_model(df)
    return model, acc

# --------------------------- WEATHER (OpenWeatherMap) ---------------------------
def fetch_weather_for_city(city_name):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name},IN&appid={OPENWEATHER_API_KEY}&units=metric"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        temp = float(j["main"]["temp"])
        humidity = float(j["main"]["humidity"])
        rainfall = float(j.get("rain", {}).get("1h", 0.0))
        desc = j["weather"][0]["description"].title()
        return {"temperature": temp, "humidity": humidity, "rainfall": rainfall, "description": desc}
    except Exception as e:
        return {"error": str(e)}

# --------------------------- UTILS: suitability and suggestions ---------------------------
def compute_suitability_percent(input_vals, ideal_range):
    # For each parameter, compute closeness score 0..1 then average to percent
    scores = []
    for k in ["N","P","K","ph","temperature","humidity","rainfall"]:
        if k not in ideal_range:
            continue
        low, high = ideal_range[k]
        val = float(input_vals.get(k, 0))
        # If within range => score 1
        if low <= val <= high:
            score = 1.0
        else:
            # linear fall-off: distance normalized to range width*2
            width = max(1.0, high - low)
            score = max(0.0, 1.0 - (abs(val - (low+high)/2) / (width*2)))
        scores.append(score)
    if not scores:
        return 0.0
    return round(float(np.mean(scores) * 100), 1)

def improvement_suggestions(input_vals, ideal_range):
    suggestions = []
    for k in ["N","P","K","ph","temperature","humidity","rainfall"]:
        if k not in ideal_range:
            continue
        low, high = ideal_range[k]
        val = float(input_vals.get(k, 0))
        if low <= val <= high:
            continue
        if k == "ph":
            if val < low:
                suggestions.append(t(f"Increase soil pH from {val} → target {low:.1f} (apply agricultural lime).",
                                     f"मातीचा pH {val} वरून {low:.1f} पर्यंत वाढवा (शेतीवाला चून लावा)."))
            else:
                suggestions.append(t(f"Decrease soil pH from {val} → target {high:.1f} (add elemental sulfur / organic matter).",
                                     f"मातीचा pH {val} वरून {high:.1f} पर्यंत कमी करा (सल्फर किंवा सेंद्रिय पदार्थ जोडा)."))
        elif k in ["N","P","K"]:
            if val < low:
                # Use fragment translations for the consistent phrasing in Marathi
                if get_lang() == "मराठी":
                    suggestions.append(f"{k} वाढवा: सध्या {val} → लक्ष्य {low} (शिफारस केलेले खत वापरा ज्यात {k} असलेले).")
                else:
                    suggestions.append(f"Increase {k} from {val} → at least {low} (apply recommended fertilizer containing {k}).")
            else:
                suggestions.append(t(f"{k} is high ({val}) — avoid adding more {k}-rich fertilizer; consider balanced fertilizer.",
                                     f"{k} जास्त आहे ({val}) — अधिक {k} खत देऊ नका; संतुलित खत वापरा."))
        elif k in ["temperature","humidity","rainfall"]:
            # can't directly change climate; give mitigations
            if k == "temperature":
                if val < low:
                    suggestions.append(t(f"Temperature ({val}°C) is low for crop — consider planting in warmer window/season or use greenhouse techniques.",
                                         f"तापमान ({val}°C) कमी आहे — उबदार हंगामात पेरणी करा किंवा ग्रीनहाऊस वापरा."))
                else:
                    suggestions.append(t(f"Temperature ({val}°C) is high — consider shade/netting or irrigation scheduling.",
                                         f"तापमान ({val}°C) जास्त आहे — सावली/नेटिंग किंवा पाणी व्यवस्थापन करा."))
            elif k == "humidity":
                if val < low:
                    suggestions.append(t(f"Humidity ({val}%) is low — consider irrigation, mulching to increase micro-humidity.",
                                         f"आर्द्रता ({val}%) कमी आहे — सिंचन किंवा मल्चिंग करा."))
                else:
                    suggestions.append(t(f"Humidity ({val}%) is high — ensure good drainage and disease management.",
                                         f"आर्द्रता ({val}%) जास्त आहे — उत्तम ड्रेनेज आणि रोग नियंत्रण करा."))
            elif k == "rainfall":
                if val < low:
                    suggestions.append(t(f"Rainfall ({val} mm) is low — consider irrigation planning or select drought-tolerant varieties.",
                                         f"पर्जन्य ({val} मिमी) कमी आहे — सिंचन नियोजन करा किंवा दुष्काळ-प्रतिरोधक जाती वापरा."))
                else:
                    suggestions.append(t(f"Rainfall ({val} mm) is high — consider drainage to avoid waterlogging.",
                                         f"पर्जन्य ({val} मिमी) जास्त आहे — ड्रेनेज सुनिश्चित करा."))
    return suggestions

# --------------------------- PDF report generation (with graphs) ---------------------------
def create_plots_bytes(input_vals, ideal, crop_name):
    # Bar chart: Input vs Ideal (for N,P,K,ph,temp,humidity,rainfall)
    keys = ["N","P","K","ph","temperature","humidity","rainfall"]
    input_vals_list = [float(input_vals.get(k, 0)) for k in keys]
    # For ideal, use mid-point of ideal range for plotting
    ideal_mid = [ (ideal[k][0] + ideal[k][1]) / 2 if k in ideal else 0 for k in keys ]

    # Create bar chart
    fig1, ax = plt.subplots(figsize=(8,4))
    x = np.arange(len(keys))
    width = 0.35
    ax.bar(x - width/2, input_vals_list, width, label=t("Your Land","तुमचे शेत"))
    ax.bar(x + width/2, ideal_mid, width, label=t("Ideal (midpoint)","आदर्श (मध्यम)"))
    ax.set_xticks(x)
    ax.set_xticklabels([t("N","N"), t("P","P"), t("K","K"), t("pH","pH"),
                        t("Temp","तापमान"), t("Hum","आर्द्रता"), t("Rain","पर्जन्य")])
    ax.set_ylabel(t("Value","मूल्य"))
    ax.set_title(t(f"Input vs Ideal (crop: {crop_name})", f"इनपुट व आदर्श (पिक: {crop_name})"))
    ax.legend()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    buf1 = BytesIO()
    plt.tight_layout()
    fig1.savefig(buf1, format="png")
    plt.close(fig1)
    buf1.seek(0)

    # Pie chart for suitability (using compute_suitability_percent)
    suit = compute_suitability_percent(input_vals, ideal)
    fig2, ax2 = plt.subplots(figsize=(4,4))
    labels = [f"{t('Suitable','योग्य')} {suit}%", t("Remaining","बाकी")] if get_lang()=="मराठी" else [f"Suitable {suit}%", "Remaining"]
    ax2.pie([suit, 100 - suit], labels=labels, autopct="%1.0f%%")
    ax2.set_title(t("Suitability","योग्यता"))
    buf2 = BytesIO()
    plt.tight_layout()
    fig2.savefig(buf2, format="png")
    plt.close(fig2)
    buf2.seek(0)

    return buf1, buf2

def generate_pdf_bytes(result_record, input_vals, ideal_range, crop_name):
    plot1_buf, plot2_buf = create_plots_bytes(input_vals, ideal_range, crop_name)
    # Convert to PIL Images
    img1 = Image.open(plot1_buf)
    img2 = Image.open(plot2_buf)

    packet = BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, t("Smart Crop & Land Improvement Report", "स्मार्ट पिक व जमीन सुधारणा अहवाल"))
    y -= 24
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"{t('Generated','निर्मित')}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 18
    c.drawString(margin, y, f"{t('Crop','पिक')}: {translate_crop_name(result_record.get('predicted_crop'))}")
    y -= 18
    c.drawString(margin, y, f"{t('District / City','जिला / शहर')}: {result_record.get('city','-')}")
    y -= 22

    # Input summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, t("Input Summary:","इनपुट सारांश:"))
    y -= 14
    c.setFont("Helvetica", 10)
    for k in ["N","P","K","ph","temperature","humidity","rainfall"]:
        label = k
        if k == "temperature":
            label = t("Temperature (°C)","तापमान (°C)")
        elif k == "humidity":
            label = t("Humidity (%)","आर्द्रता (%)")
        elif k == "rainfall":
            label = t("Rainfall (mm)","पर्जन्य (मिमी)")
        c.drawString(margin + 6, y, f"{label}: {input_vals.get(k)}")
        y -= 12

    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, t("Predicted / Suggested:","अनुमान / शिफारस:"))
    y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(margin + 6, y, f"{t('Predicted Crop','अनुमानित पिक')}: {translate_crop_name(result_record.get('predicted_crop'))}")
    y -= 12
    c.drawString(margin + 6, y, f"{t('Suitability','योग्यता')}: {result_record.get('suitability')}%")
    y -= 12
    c.drawString(margin + 6, y, f"{t('Recommendations','शिफारसी')}:")
    y -= 12
    for rec in result_record.get("recommendations", []):
        # wrap long text
        text = rec
        words = text.split()
        cur = ""
        for w in words:
            if len(cur) + len(w) + 1 <= 90:
                cur += w + " "
            else:
                c.drawString(margin + 12, y, cur.strip())
                y -= 12
                cur = w + " "
        if cur:
            c.drawString(margin + 12, y, cur.strip())
            y -= 12
        y -= 4

    # Add plots (ensure enough space - new page if needed)
    c.showPage()
    # page 2: plots
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, height - margin, t("Charts","चार्ट"))
    # draw images scaled
    img1_reader = ImageReader(img1)
    img2_reader = ImageReader(img2)
    # Draw first big plot
    c.drawImage(img1_reader, margin, height - margin - 300, width=width - 2*margin, height=300, preserveAspectRatio=True)
    c.showPage()
    # page 3: pie
    c.drawImage(img2_reader, margin + 100, height/2 - 100, width=300, height=300, preserveAspectRatio=True)
    c.save()
    packet.seek(0)
    return packet.read()

# --------------------------- STREAMLIT UI ---------------------------
def main():
    # minimal language selector in top-right (keeps layout unchanged)
    top_left, _, top_right = st.columns([6,1,1])
    with top_right:
        lang_choice = st.selectbox("", ["English", "मराठी"], index=0 if get_lang()=="English" else 1)
        st.session_state["lang"] = lang_choice

    st.title("🌾 Smart Crop & Land Improvement (2025)")
    st.markdown(t("**Enter your farm data** (or fetch live weather for your district) — get crop suggestions and a PDF report with graphs.",
                  "**तुमची शेती माहिती भरा** (किंवा आपल्या जिल्ह्याचे हवामान घ्या) — पिक सूचना आणि PDF अहवाल मिळवा."))

    # left column: inputs
    col1, col2 = st.columns([2,3])
    with col1:
        st.subheader(t("Farm Location & Weather","शेती स्थान व हवामान"))
        if get_lang() == "मराठी":
            # show Marathi district names but keep internal 'city' as English key
            options_mr = [DISTRICTS_MR_MAP.get(d, d) for d in DISTRICTS]
            sel_mr = st.selectbox(t("Select District / City (Maharashtra)","जिल्हा / शहर निवडा (महाराष्ट्र)"),
                                  options_mr, index=options_mr.index(DISTRICTS_MR_MAP.get("Pune","पुणे")) if "Pune" in DISTRICTS else 0)
            # map selected Marathi back to English key
            city = next((en for en,mr in DISTRICTS_MR_MAP.items() if mr == sel_mr), sel_mr)
        else:
            city = st.selectbox(t("Select District / City (Maharashtra)","जिल्हा / शहर निवडा (महाराष्ट्र)"),
                                DISTRICTS, index=DISTRICTS.index("Pune") if "Pune" in DISTRICTS else 0)

        if st.button(t("Fetch Live Weather for Selected District","निवडलेल्या जिल्ह्यासाठी हवामान मिळवा")):
            with st.spinner(t("Fetching weather...","हवामान मिळत आहे...")):
                w = fetch_weather_for_city(city)
                if "error" in w:
                    st.error(t("Weather fetch error: ","हवामान मिळवताना त्रुटी: ") + w["error"])
                else:
                    st.success(t("Weather fetched: ","हवामान मिळाले: ") + w["description"])
                    # store into session_state so the form picks defaults
                    st.session_state["weather_override"] = w

        # Show fetched weather
        wov = st.session_state.get("weather_override", {})
        if wov:
            st.metric(t("🌡 Temperature (°C)","🌡 तापमान (°C)"), wov.get("temperature"))
            st.metric(t("💧 Humidity (%)","💧 आर्द्रता (%)"), wov.get("humidity"))
            st.metric(t("🌧 Rainfall (mm)","🌧 पर्जन्य (मिमी)"), wov.get("rainfall"))

        st.markdown("---")
        st.subheader(t("Soil & Field Inputs","माती व क्षेत्र माहिती"))
        # Allow farmer to still manually enter N,P,K,pH. Default values can be overridden by session_state or kept from earlier
        N = st.number_input(t("Nitrogen (N)","नायट्रोजन (N)"), min_value=0.0, max_value=500.0, value=float(st.session_state.get("N", 50.0)))
        P = st.number_input(t("Phosphorus (P)","फॉस्फरस (P)"), min_value=0.0, max_value=500.0, value=float(st.session_state.get("P", 40.0)))
        K = st.number_input(t("Potassium (K)","पोटॅशियम (K)"), min_value=0.0, max_value=500.0, value=float(st.session_state.get("K", 40.0)))
        ph = st.number_input(t("Soil pH","मातीचा pH"), min_value=0.0, max_value=14.0, value=float(st.session_state.get("ph", 6.5)))
        # For temperature/humidity/rainfall present them but allow override
        temp_default = wov.get("temperature", st.session_state.get("temperature", 25.0))
        hum_default = wov.get("humidity", st.session_state.get("humidity", 70.0))
        rain_default = wov.get("rainfall", st.session_state.get("rainfall", 50.0))
        temperature = st.number_input(t("Temperature (°C)","तापमान (°C)"), value=float(temp_default))
        humidity = st.number_input(t("Humidity (%)","आर्द्रता (%)"), value=float(hum_default))
        rainfall = st.number_input(t("Recent rainfall (mm)","सद्य पर्जन्य (मिमी)"), value=float(rain_default))

        st.markdown("---")
        # optional: farmer can request a specific crop to attempt to grow
        st.subheader(t("Desired Crop (optional)","इच्छित पिक (ऐच्छिक)"))
        no_pref = t("(No preference)","(प्राथमिकता नाही)")
        crop_keys = sorted(list(IDEAL_RANGES.keys()))
        if get_lang() == "मराठी":
            # show Marathi crop names in selectbox
            crop_options_mr = [CROP_TRANSLATIONS.get(k, k) for k in crop_keys]
            crop_options = [no_pref] + crop_options_mr
            sel_crop_display = st.selectbox(t("Choose crop you'd like to grow (optional)","तुम्हाला जे पिक लावायचे आहे ते निवडा (ऐच्छिक)"), crop_options)
            if sel_crop_display == no_pref:
                desired_crop = no_pref
            else:
                # map back Marathi display to English key
                desired_crop = next((k for k in crop_keys if CROP_TRANSLATIONS.get(k,k) == sel_crop_display), sel_crop_display)
        else:
            desired_crop = st.selectbox(t("Choose crop you'd like to grow (optional)","तुम्हाला जे पिक लावायचे आहे ते निवडा (ऐच्छिक)"),
                                       [no_pref] + crop_keys)

        st.markdown(t("If you select a crop, the system will compare your inputs vs ideal requirements and suggest improvements.",
                      "जर तुम्ही पिक निवडले तर सिस्टम तुमच्या इनपुटची आदर्श आवश्यकता बरोबर तपासून सुधारणा सुचवेल."))

    # right column: action & results
    with col2:
        st.subheader(t("Actions & Results","क्रिया व परिणाम"))
        model, model_acc = load_or_train_model()
        if model_acc is not None:
            st.info(t(f"Trained model (sample) accuracy: {model_acc:.2f}", f"प्रशिक्षित मॉडेलची अचूकता: {model_acc:.2f}"))

        if st.button(t("Predict Best Crop (based on inputs)","सर्वोत्तम पिक अंदाज (इनपुटनुसार)")):
            features = {"N": N, "P": P, "K": K, "temperature": temperature, "humidity": humidity, "ph": ph, "rainfall": rainfall}
            pred = model.predict(pd.DataFrame([features]))[0]
            pred_display = translate_crop_name(pred)
            st.success(t(f"Recommended crop: **{pred.upper()}**", f"शिफारस: **{pred_display}**"))

            # compute suitability against ideal for predicted crop if available
            ideal = IDEAL_RANGES.get(pred, None)
            if ideal:
                suit = compute_suitability_percent(features, ideal)
                st.metric(t("Suitability (%) for predicted crop","पिकासाठी अनुकूलता (%)"), f"{suit}%")
                recs = improvement_suggestions(features, ideal)
                if recs:
                    st.subheader(t("Suggestions to improve land for this crop","या पिकासाठी जमिन सुधारण्यासाठी सूचना"))
                    for r in recs:
                        st.write("• " + r)
                else:
                    st.write(t("Your land matches ideal ranges for this crop. Good job!","तुमची जमीन या पिकासाठी अनुकूल आहे. छान!"))
            else:
                st.warning(t("No ideal range data available for predicted crop.","अनुमानित पिकासाठी आदर्श श्रेणी माहिती उपलब्ध नाही."))

            # store result in session for download/report
            st.session_state["prediction_result"] = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "city": city,
                "predicted_crop": pred,
                "features": features,
                "suitability": suit if ideal else None,
                "recommendations": recs if ideal else []
            }

        st.markdown("---")
        st.subheader(t("Or: Evaluate a Desired Crop","किंवा: इच्छित पिकाचे मूल्यांकन करा"))
        if desired_crop != no_pref:
            if st.button(t("Evaluate Desired Crop","इच्छित पिकाचे मूल्यांकन करा")):
                features = {"N": N, "P": P, "K": K, "temperature": temperature, "humidity": humidity, "ph": ph, "rainfall": rainfall}
                crop_key = desired_crop
                # if Marathi mode may have mapped to english key already above
                ideal = IDEAL_RANGES.get(crop_key)
                if ideal is None:
                    st.warning(t("No ideal range data available for this crop.","या पिकासाठी आदर्श श्रेणी माहिती उपलब्ध नाही."))
                else:
                    suit = compute_suitability_percent(features, ideal)
                    display_crop = translate_crop_name(crop_key)
                    st.success(t(f"Suitability for {crop_key}: {suit}%", f"{display_crop} साठी अनुकूलता: {suit}%"))
                    recs = improvement_suggestions(features, ideal)
                    if recs:
                        st.subheader(t("Improvements needed to grow this crop","या पिकाच्या लागवडीसाठी आवश्यक सुधारणा"))
                        for r in recs:
                            st.write("• " + r)
                    else:
                        st.write(t("Your land is suitable for this crop.","तुमची जमीन या पिकासाठी योग्य आहे."))
                    st.session_state["prediction_result"] = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "city": city,
                        "predicted_crop": crop_key,
                        "features": features,
                        "suitability": suit,
                        "recommendations": recs
                    }

        st.markdown("---")
        # Download CSV and PDF
        pres = st.session_state.get("prediction_result")
        if pres:
            st.subheader(t("Download / Save Results","परिणाम डाउनलोड / जतन करा"))
            # CSV
            if get_lang() == "मराठी":
                csv_record = {
                    "टाइमस्टँप": pres["timestamp"], "जिला": pres["city"],
                    "नायट्रोजन (N)": pres["features"].get("N"), "फॉस्फरस (P)": pres["features"].get("P"),
                    "पोटॅशियम (K)": pres["features"].get("K"), "pH": pres["features"].get("ph"),
                    "तापमान (°C)": pres["features"].get("temperature"), "आर्द्रता (%)": pres["features"].get("humidity"),
                    "पर्जन्य (मिमी)": pres["features"].get("rainfall"),
                    "अनुमानित पिक": translate_crop_name(pres["predicted_crop"]),
                    "योग्यता": pres.get("suitability"), "शिफारसी": " | ".join(pres.get("recommendations", []))
                }
            else:
                csv_record = {
                    "timestamp": pres["timestamp"], "city": pres["city"], "Nitrogen (N)": pres["features"].get("N"),
                    "Phosphorus (P)": pres["features"].get("P"), "Potassium (K)": pres["features"].get("K"), "pH": pres["features"].get("ph"),
                    "Temperature (°C)": pres["features"].get("temperature"), "Humidity (%)": pres["features"].get("humidity"), "Rainfall (mm)": pres["features"].get("rainfall"),
                    "predicted_crop": pres["predicted_crop"], "suitability": pres.get("suitability"), "recommendations": " | ".join(pres.get("recommendations", []))
                }
            csv_df = pd.DataFrame([csv_record])
            csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
            st.download_button(t("📥 Download CSV","📥 CSV डाउनलोड करा"), data=csv_bytes, file_name=f"{pres['city']}_prediction_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv", mime="text/csv")

            # PDF (generate)
            if st.button(t("📄 Generate PDF Report","📄 PDF अहवाल तयार करा")):
                ideal = IDEAL_RANGES.get(pres["predicted_crop"].lower(), None)
                if not ideal:
                    st.warning(t("No ideal ranges to plot for this crop; PDF will include textual summary.","आदर्श श्रेणी उपलब्ध नाही; PDF मध्ये मजकूर सारांश असेल."))
                    pdf_bytes = generate_pdf_bytes(pres, pres["features"], pres["features"], pres["predicted_crop"])
                else:
                    pdf_bytes = generate_pdf_bytes(pres, pres["features"], ideal, pres["predicted_crop"])
                st.download_button(t("📥 Download PDF Report","📥 PDF अहवाल डाउनलोड करा"), data=pdf_bytes, file_name=f"{pres['city']}_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf", mime="application/pdf")

    # footer
    st.markdown("---")
    st.markdown(f"<small style='color:gray'>{t('Built by Sarthak Dhumal','डिझाइन: सार्थक धुमल')}</small>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
