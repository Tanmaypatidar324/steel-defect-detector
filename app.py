import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Steel Defect Detector", layout="wide")
st.title("AI Steel Surface Defect Detector")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

if "total" not in st.session_state:
    st.session_state.total = 0
    st.session_state.defects = {}

file = st.file_uploader("Upload photo of steel", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    r = model.predict(img, imgsz=224)[0]
    label = model.names[r.probs.top1]
    conf = float(r.probs.top1conf)

    col1, col2 = st.columns(2)
    col1.image(img, caption="Uploaded photo", use_container_width=True)

    with col2:
        if conf >= 0.6:
            st.error(f"DEFECT MILA: {label}")
            st.write(f"Confidence: {conf:.0%}")
        else:
            st.warning("Uncertain: manual inspection ke liye bhejo")
            st.write(f"Sabse possible: {label} ({conf:.0%})")

    if st.button("Record this item"):
        st.session_state.total += 1
        st.session_state.defects[label] = st.session_state.defects.get(label, 0) + 1

st.divider()
st.metric("Total checked", st.session_state.total)
if st.session_state.defects:
    st.bar_chart(st.session_state.defects)