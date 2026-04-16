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

    W, H = 900, 650
    cert = Image.new("RGB", (W, H), (252, 250, 245))
    draw = ImageDraw.Draw(cert)

    # =========================
    # BORDER
    # =========================
    draw.rectangle([(25, 25), (W - 25, H - 25)], outline=(180, 160, 120), width=3)
    draw.rectangle([(45, 45), (W - 45, H - 45)], outline=(210, 200, 170), width=2)

    # =========================
    # FONTS
    # =========================
    try:
        title_font = ImageFont.truetype("timesbd.ttf", 90)
        subtitle_font = ImageFont.truetype("times.ttf", 50)
        label_font = ImageFont.truetype("times.ttf", 40)
        value_font = ImageFont.truetype("timesbd.ttf", 48)
    except:
        title_font = subtitle_font = label_font = value_font = ImageFont.load_default()

    def center(text, y, font, fill):
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((W - w) / 2, y), text, font=font, fill=fill)

    # =========================
    # HEADER
    # =========================
    center("CERTIFICATE OF DIAGNOSTIC EVALUATION", 55, title_font, (70, 60, 50))
    center("Official Hair Health Assessment Report", 140, subtitle_font, (120, 110, 100))

    # =========================
    # USER IMAGE
    # =========================
    user_img = ImageOps.fit(user_img, (340, 340))
    cert.paste(user_img, (280, 170))

    # =========================
    # BADGE
    # =========================
    badge = Image.open(badge_path).convert("RGBA")
    badge = ImageOps.fit(badge, (150, 150))

    mask = Image.new("L", (150, 150), 0)
    m = ImageDraw.Draw(mask)
    m.ellipse((0, 0, 150, 150), fill=255)
    badge.putalpha(mask)

    cert.paste(badge, (700, 440), badge)

    # =========================
    # RESULT
    # =========================
    result = class_name.replace("_", " ").title()
    confidence = f"{conf_score * 100:.2f}%"

    color = (160, 40, 40) if class_name != "no_alopecia" else (40, 120, 80)

    draw.text((170, 540), "Diagnosis:", fill=(90, 80, 70), font=label_font)
    draw.text((420, 535), result, fill=color, font=value_font)

    draw.text((170, 590), "Confidence:", fill=(90, 80, 70), font=label_font)
    draw.text((420, 585), confidence, fill=(60, 60, 60), font=value_font)

    # =========================
    # FOOTER
    # =========================
    footer = "This document is automatically generated and digitally certified."

    bbox = draw.textbbox((0, 0), footer, font=label_font)
    text_w = bbox[2] - bbox[0]

    draw.text(
        ((W - text_w) / 2, H - 40),
        footer,
        fill=(140, 130, 120),
        font=label_font
    )

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
