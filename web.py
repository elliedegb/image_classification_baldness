import streamlit as st
import pickle
import base64
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
from skimage.transform import resize
from skimage.feature import hog
from skimage.color import rgb2gray
import io

# =========================
# BACKGROUND
# =========================
def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    /* DARK OVERLAY */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.55);
        z-index: 0;
    }}

    .main {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """, unsafe_allow_html=True)


set_background("./bg.png")


# =========================
# CLASSIFY
# =========================
def classify(image, model, class_names):

    image = np.array(image)
    image = resize(image, (128, 128), anti_aliasing=True)
    image_gray = rgb2gray(image)

    features = hog(
        image_gray,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    data = features.reshape(1, -1)

    pred = model.predict(data)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(data)
        conf = np.max(prob)
    else:
        conf = 1.0

    return class_names[pred[0]], conf


# =========================
# CERTIFICATE
# =========================
def create_certificate(user_img, badge_path, class_name, conf_score):

    W, H = 1400, 1000
    cert = Image.new("RGB", (W, H), (252, 250, 245))
    draw = ImageDraw.Draw(cert)

    # BORDER
    draw.rectangle([(40, 40), (W - 40, H - 40)], outline=(180, 160, 120), width=6)
    draw.rectangle([(80, 80), (W - 80, H - 80)], outline=(210, 200, 170), width=4)

    # FONTS
    try:
        title_font = ImageFont.truetype("timesbd.ttf", 90)
        subtitle_font = ImageFont.truetype("times.ttf", 50)
        label_font = ImageFont.truetype("times.ttf", 40)
        value_font = ImageFont.truetype("timesbd.ttf", 55)
    except:
        title_font = subtitle_font = label_font = value_font = ImageFont.load_default()

    def center(text, y, font, fill):
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((W - w) / 2, y), text, font=font, fill=fill)

    # HEADER
    center("CERTIFICATE OF DIAGNOSTIC EVALUATION", 120, title_font, (70, 60, 50))
    center("Official Hair Health Assessment Report", 240, subtitle_font, (120, 110, 100))

    # USER IMAGE
    user_img = ImageOps.fit(user_img, (500, 500))
    cert.paste(user_img, (450, 320))

    # BADGE
    badge = Image.open(badge_path).convert("RGBA")
    badge = ImageOps.fit(badge, (220, 220))

    mask = Image.new("L", (220, 220), 0)
    m = ImageDraw.Draw(mask)
    m.ellipse((0, 0, 220, 220), fill=255)
    badge.putalpha(mask)

    cert.paste(badge, (1050, 700), badge)

    # GOLD SEAL
    seal_size = 200
    seal = Image.new("RGBA", (seal_size, seal_size), (0, 0, 0, 0))
    seal_draw = ImageDraw.Draw(seal)

    center_s = seal_size // 2

    for r in range(center_s, 0, -1):
        color = (255, 215 - int(60 * (r / center_s)), 80)
        seal_draw.ellipse((center_s - r, center_s - r, center_s + r, center_s + r), fill=color)

    seal_draw.ellipse((10, 10, seal_size - 10, seal_size - 10), outline=(120, 90, 30), width=4)
    seal_draw.ellipse((30, 30, seal_size - 30, seal_size - 30), outline=(150, 120, 40), width=2)

    try:
        seal_font = ImageFont.truetype("timesbd.ttf", 24)
    except:
        seal_font = ImageFont.load_default()

    seal_draw.text((40, 80), "OFFICIAL", fill=(90, 70, 20), font=seal_font)
    seal_draw.text((55, 110), "SEAL", fill=(90, 70, 20), font=seal_font)

    seal = seal.rotate(-12, expand=True)
    shadow = seal.copy().filter(ImageFilter.GaussianBlur(5))

    cert.paste(shadow, (720, 310), shadow)
    cert.paste(seal, (700, 290), seal)

    # RESULT
    result = class_name.replace("_", " ").title()
    confidence = f"{conf_score * 100:.2f}%"

    color = (160, 40, 40) if class_name != "no_alopecia" else (40, 120, 80)

    draw.text((300, 780), "Diagnosis:", fill=(90, 80, 70), font=label_font)
    draw.text((550, 770), result, fill=color, font=value_font)

    draw.text((300, 850), "Confidence:", fill=(90, 80, 70), font=label_font)
    draw.text((550, 840), confidence, fill=(60, 60, 60), font=value_font)

    # SIGNATURES
    line_y = 720
    draw.line((200, line_y, 450, line_y), fill=(120, 110, 100), width=3)
    draw.text((240, line_y + 10), "Certified Analyst", fill=(90, 80, 70), font=label_font)

    draw.line((650, line_y, 900, line_y), fill=(120, 110, 100), width=3)
    draw.text((690, line_y + 10), "AI Evaluation System", fill=(90, 80, 70), font=label_font)

    # FOOTER
    footer = "This document is automatically generated and digitally certified."
    bbox = draw.textbbox((0, 0), footer, font=label_font)
    w = bbox[2] - bbox[0]

    draw.text(((W - w) / 2, H - 80), footer, fill=(140, 130, 120), font=label_font)

    return cert



# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.p", "rb"))
class_names = ["alopecia", "no_alopecia", "receding_hairline"]


# =========================
# UI
# =========================
st.title("🧑‍🦲 Baldness Certificate Generator")

# =========================
# EXPLANATION (USER-FRIENDLY)
# =========================
with st.expander("🧠 How does this work?"):
    st.markdown("""
**The Baldness Certificate Generator** uses machine learning to analyze your photo and identify patterns related to hair conditions.

When you upload an image, the system doesn’t actually *see* it like a human. Instead, it converts the image into numbers that describe important visual features—especially edges and shapes. This helps focus on things like hairlines, density, and texture.

To make the analysis consistent, the image is resized and simplified (turned into grayscale). Then, a technique called **feature extraction** summarizes the image into a compact representation that highlights patterns related to hair structure.

These features are passed into a trained machine learning model, which has learned from many labeled examples such as **Alopecia**, **Receding Hairline**, and **No Alopecia**.

Based on that, the model predicts which category your image most closely matches.

Finally, you’ll see your result, a confidence score, and receive your personalized **certificate with a badge** 🎖️.
""")

file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])


# =========================
# MAIN
# =========================
if file is not None:

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Input image", width=300)

    class_name, conf_score = classify(image, model, class_names)

    # ✨ PRETTY NAME
    pretty_name = class_name.replace("_", " ").title()

    st.subheader("🎯 Prediction Result")
    st.write(f"Class: **{pretty_name}**")
    st.write(f"Confidence: **{conf_score*100:.2f}%**")

    if class_name == "alopecia":
        badge_path = "./alopecia.jpg"
    elif class_name == "receding_hairline":
        badge_path = "./receding.jpg"
    else:
        badge_path = "./no_alopecia.png"

    cert = create_certificate(image, badge_path, class_name, conf_score)

    st.subheader("🎖️ Your Certificate")
    st.image(cert, use_container_width=True)
    # =========================
    # DOWNLOAD BUTTON
    # =========================
    buf = io.BytesIO()
    cert.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="💾 Download Certificate",
        data=byte_im,
        file_name="baldness_certificate.png",
        mime="image/png"
        )
