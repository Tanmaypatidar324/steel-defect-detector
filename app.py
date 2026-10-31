import html
import os
import re
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from google import genai
from google.genai import types

# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="SteelScan AI", page_icon="🔍", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f1f5f9; }
    .stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2,
    .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp .stMarkdown,
    .stApp div[data-testid="stSpinner"] * { color: #1e293b; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(120deg, #0b1d33 0%, #12355b 60%, #1d4e89 100%);
        padding: 30px 36px; border-radius: 14px; margin-bottom: 26px;
        box-shadow: 0 6px 20px rgba(11, 29, 51, 0.25);
    }
    .hero, .hero * { color: #ffffff !important; }
    .hero h1 { margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: 0.3px; }
    .hero p { margin: 6px 0 0 0; font-size: 1rem; color: #cbd5e1 !important; }
    .chips { margin-top: 14px; }
    .chips span {
        display: inline-block; background: rgba(255,255,255,0.14);
        padding: 4px 12px; border-radius: 4px; margin: 0 8px 0 0;
        font-size: 0.8rem; color: #ffffff !important;
    }

    .section-title {
        font-size: 0.8rem; font-weight: 700; letter-spacing: 1px;
        text-transform: uppercase; color: #475569 !important; margin: 4px 0 10px 0;
    }
    .card {
        background: #ffffff; padding: 22px 26px; border-radius: 12px;
        border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
        margin-bottom: 14px;
    }
    .card, .card p, .card b { color: #1e293b !important; }
    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 4px;
        font-weight: 700; font-size: 0.75rem; letter-spacing: 0.8px;
        color: #ffffff !important;
    }
    .big-label { font-size: 1.8rem; font-weight: 800; margin: 10px 0 8px 0; }
    .report {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-left: 5px solid #1d4e89; padding: 18px 22px; border-radius: 8px;
        line-height: 1.7;
    }
    .report, .report b { color: #1e293b !important; }
    .notice { padding: 12px 16px; border-radius: 8px; margin: 8px 0; font-size: 0.95rem; }
    .notice-warn { background: #fffbeb; border: 1px solid #fcd34d; color: #92400e !important; }
    .notice-ok { background: #ecfdf5; border: 1px solid #6ee7b7; color: #065f46 !important; }
    .notice-info { background: #eff6ff; border: 1px solid #93c5fd; color: #1e3a8a !important; }

    .bar-bg { background: #e2e8f0; border-radius: 6px; height: 10px; width: 100%; }
    .bar-fill { height: 10px; border-radius: 6px; }

    .stat {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 18px; text-align: center; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    }
    .stat .num { font-size: 2.1rem; font-weight: 800; }
    .stat .lbl { color: #64748b !important; font-size: 0.85rem; }

    .row { display: flex; align-items: center; margin: 8px 0; }
    .row .name { width: 170px; font-size: 0.9rem; color: #1e293b !important; }
    .row .track { flex: 1; background: #e2e8f0; border-radius: 4px; height: 16px; }
    .row .fill { background: #1d4e89; height: 16px; border-radius: 4px; }
    .row .cnt { width: 40px; text-align: right; font-weight: 700; color: #1e293b !important; }
    table.log { width: 100%; border-collapse: collapse; background: #ffffff;
        border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
    table.log th { background: #0b1d33; color: #ffffff !important; padding: 10px 12px;
        text-align: left; font-size: 0.85rem; }
    table.log td { padding: 9px 12px; border-top: 1px solid #e2e8f0;
        color: #1e293b !important; font-size: 0.9rem; }

    .stApp .stButton > button {
        background: #1d4e89; color: #ffffff !important; border: none;
        border-radius: 8px; padding: 8px 22px; font-weight: 600;
    }
    .stApp .stButton > button:hover { background: #12355b; color: #ffffff !important; }
    .stApp .stButton > button p { color: #ffffff !important; }

    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff; border: 2px dashed #94a3b8; border-radius: 12px;
    }
    [data-testid="stFileUploaderDropzone"] * { color: #334155 !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: #1d4e89; color: #ffffff !important; border: none;
    }
    [data-testid="stFileUploaderDropzone"] button * { color: #ffffff !important; }
    [data-testid="stFileUploaderFile"] * { color: #1e293b !important; }
    [data-testid="stCameraInput"] button { background: #1d4e89; color: #ffffff !important; }
    [data-testid="stCameraInput"] button * { color: #ffffff !important; }

    section[data-testid="stSidebar"] { background: #0b1d33; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] input[type="password"],
    section[data-testid="stSidebar"] input[type="text"] {
        background: #14304f !important; color: #ffffff !important;
        border: 1px solid #2c4a6e !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: #1d4e89; color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ knowledge
SEV_COLOR = {"High": "#dc2626", "Medium": "#d97706", "Low": "#059669", "Unknown": "#64748b"}
SEV_BGR = {"High": (38, 38, 220), "Medium": (6, 119, 217), "Low": (105, 150, 5),
           "Unknown": (139, 116, 100)}

DEFECT_INFO = {
    "crazing": ("Medium", "A fine network of surface cracks on the steel.",
                "Check the cooling process and re-inspect the batch."),
    "inclusion": ("High", "Foreign particles trapped in the steel surface.",
                  "Reject the item and check raw material quality."),
    "patches": ("Medium", "Uneven or discoloured patches on the surface.",
                "Send for re-polishing and review the surface treatment."),
    "pitted_surface": ("High", "Small pits or holes, usually caused by corrosion.",
                       "Reject the item and trace the source of corrosion."),
    "rolled-in_scale": ("Medium", "Oxide scale pressed into the steel during rolling.",
                        "Clean the mill rollers and descale the line."),
    "scratches": ("Low", "Linear marks caused by friction or handling.",
                  "Re-polish the surface and inspect again."),
}


def pretty(name):
    return name.replace("_", " ").replace("-", " ").title()


def md_to_html(text):
    safe = html.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    return safe.replace("\n", "<br>")


def notice(kind, text):
    st.markdown(f'<div class="notice notice-{kind}">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------- model
@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


def looks_like_steel(arr):
    small = cv2.resize(np.asarray(arr), (128, 128)).astype(int)
    diff = (np.abs(small[:, :, 0] - small[:, :, 1]) + np.abs(small[:, :, 1] - small[:, :, 2])).mean()
    return diff < 12


def classify(source):
    r = model.predict(source, imgsz=224, verbose=False)[0]
    return model.names[r.probs.top1], float(r.probs.top1conf)


def info_for(label):
    return DEFECT_INFO.get(label, ("Unknown", "Unrecognised defect.", "Send for manual inspection."))


@st.cache_data(show_spinner=False)
def make_report(key, label, conf, severity, desc, action):
    backup = (
        f"**Defect:** {pretty(label)} detected (severity: {severity}).\n"
        f"**Cause:** {desc}\n"
        f"**Action:** {action}"
    )
    if not key:
        return backup
    try:
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=15000))
        prompt = (
            f"A steel surface inspection system detected the defect '{pretty(label)}' "
            f"with {conf:.0%} confidence. Severity: {severity}. Suggested action: {action}. "
            "Write a short professional inspection report in English, under 80 words, "
            "using exactly three bold headings: **Defect**, **Cause**, **Action**. "
            "Put each heading on its own line followed by one or two simple sentences."
        )
        resp = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
        return resp.text
    except Exception:
        return backup


def record_item(label, severity, conf):
    st.session_state.total += 1
    name = pretty(label)
    st.session_state.defects[name] = st.session_state.defects.get(name, 0) + 1
    st.session_state.history.append(
        {"item": st.session_state.total, "defect": name, "severity": severity, "conf": f"{conf:.0%}"}
    )


# ------------------------------------------------------------------- session
if "total" not in st.session_state:
    st.session_state.total = 0
    st.session_state.defects = {}
    st.session_state.history = []

# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## SteelScan AI")
    api_key = st.text_input("Gemini API key (optional)", type="password")
    st.caption("Add a key to get AI-written inspection reports.")
    steel_filter = st.checkbox("Reject non-steel images", value=True)
    st.caption("Turn this off if a live camera demo is being rejected.")
    st.markdown("---")
    st.markdown("### Workflow")
    st.markdown(
        "1. Choose an input: image, camera or live video\n"
        "2. AI identifies the defect type\n"
        "3. Review severity and action\n"
        "4. Defective items are flagged and logged\n"
        "5. Track statistics on the dashboard"
    )
    st.markdown("---")
    st.markdown("### Detectable defects")
    for d in DEFECT_INFO:
        st.markdown(f"- {pretty(d)}")
    st.markdown("---")
    if st.button("Reset statistics"):
        st.session_state.total = 0
        st.session_state.defects = {}
        st.session_state.history = []

# ---------------------------------------------------------------------- hero
st.markdown(
    """
    <div class="hero">
        <h1>SteelScan AI</h1>
        <p>Real-time AI quality assurance and defect detection for steel surfaces</p>
        <div class="chips">
            <span>YOLOv8 Classifier</span><span>99.3% validation accuracy</span>
            <span>6 defect classes</span><span>Live video inspection</span>
            <span>Generative AI reports</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------- single image inspection
def inspect_image(file):
    img = Image.open(file).convert("RGB")

    if steel_filter and not looks_like_steel(img):
        c1, c2 = st.columns([1, 2])
        c1.image(img, use_container_width=True)
        with c2:
            notice("warn", "This does not look like a steel surface image. "
                           "Please upload a valid steel photo.")
        return

    label, conf = classify(img)
    severity, desc, action = info_for(label)
    color = SEV_COLOR[severity]

    st.markdown('<div class="section-title">Inspection result</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.2])
    with left:
        st.image(img, caption="Inspected image", use_container_width=True)
    with right:
        if conf >= 0.6:
            st.markdown(
                f"""
                <div class="card">
                    <span class="badge" style="background:{color}">{severity.upper()} SEVERITY</span>
                    <div class="big-label" style="color:{color} !important">{pretty(label)}</div>
                    <p><b>Description:</b> {desc}</p>
                    <p><b>Recommended action:</b> {action}</p>
                    <p style="margin-bottom:6px"><b>Model confidence:</b> {conf:.0%}</p>
                    <div class="bar-bg"><div class="bar-fill" style="width:{conf*100:.0f}%;background:{color}"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="card">
                    <span class="badge" style="background:#64748b">MANUAL CHECK REQUIRED</span>
                    <div class="big-label" style="color:#64748b !important">Uncertain result</div>
                    <p>The model is not confident enough. Please send this item for manual inspection.</p>
                    <p>Closest match: <b>{pretty(label)}</b> ({conf:.0%})</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if conf >= 0.6:
        st.markdown('<div class="section-title">AI inspection report</div>', unsafe_allow_html=True)
        with st.spinner("Generating report..."):
            report = make_report(api_key, label, conf, severity, desc, action)
        st.markdown(f'<div class="report">{md_to_html(report)}</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("Record this item"):
        record_item(label, severity, conf)
        notice("ok", "Item recorded successfully.")


# ------------------------------------------------------ live video inspection
def run_live(cap):
    frame_ph = st.empty()
    stats_ph = st.empty()
    st.markdown('<div class="section-title">Flagged for inspection (latest 4)</div>',
                unsafe_allow_html=True)
    flag_ph = st.empty()

    flagged = []
    frames = 0
    last_flag = 0.0
    t0 = time.time()
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * 640 / w)))
                h, w = frame.shape[:2]
            frames += 1

            t1 = time.time()
            if steel_filter and not looks_like_steel(frame):
                text, sev, flag = "NOT STEEL", "Unknown", False
                label, conf = None, 0.0
            else:
                label, conf = classify(frame)
                sev, _, _ = info_for(label)
                if conf >= 0.6:
                    text, flag = f"{pretty(label)} {conf:.0%} | {sev}", True
                else:
                    text, sev, flag = "UNCERTAIN - manual check", "Unknown", False
            latency = (time.time() - t1) * 1000

            color = SEV_BGR[sev]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)
            cv2.rectangle(frame, (0, 0), (w, 44), color, -1)
            cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            frame_ph.image(frame, channels="BGR", use_container_width=True)

            fps = frames / max(time.time() - t0, 0.001)
            stats_ph.markdown(
                f'<div class="notice notice-info">Frames: <b>{frames}</b> &nbsp;|&nbsp; '
                f'Speed: <b>{fps:.1f} FPS</b> &nbsp;|&nbsp; '
                f'Latency: <b>{latency:.0f} ms</b> &nbsp;|&nbsp; '
                f'Flagged: <b>{len(flagged)}</b></div>',
                unsafe_allow_html=True,
            )

            if flag and time.time() - last_flag > 2:
                last_flag = time.time()
                record_item(label, sev, conf)
                flagged.append((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f"{pretty(label)} ({sev})"))
                with flag_ph.container():
                    cols = st.columns(4)
                    for c, (thumb, cap_text) in zip(cols, flagged[-4:][::-1]):
                        c.image(thumb, caption=cap_text, use_container_width=True)

            if st.session_state.get("live_src") == "Video file":
                time.sleep(0.03)
    finally:
        cap.release()
    notice("ok", "Stream ended.")


# ------------------------------------------------------------- input modes
st.markdown('<div class="section-title">Select input mode</div>', unsafe_allow_html=True)
mode = st.radio(
    "Input mode",
    ["Upload image", "Camera snapshot", "Live video stream"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "Upload image":
    file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if file:
        inspect_image(file)

elif mode == "Camera snapshot":
    shot = st.camera_input("Take a photo", label_visibility="collapsed")
    if shot:
        inspect_image(shot)

else:
    src = st.radio("Video source", ["Webcam", "Video file"], horizontal=True, key="live_src")
    video = None
    if src == "Video file":
        video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    notice("info", "Live mode classifies every frame, flags defective items automatically and "
                   "adds them to the log. AI text reports are available in image mode.")
    run = st.checkbox("Start live inspection")

    if run:
        cap = None
        if src == "Webcam":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        elif video is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(video.getvalue())
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
        else:
            notice("warn", "Please upload a video file first.")

        if cap is not None:
            if cap.isOpened():
                run_live(cap)
            else:
                notice("warn", "Could not open the video source. Check that the webcam is "
                               "connected and not used by another app.")

# ----------------------------------------------------------------- analytics
st.markdown("---")
st.markdown('<div class="section-title">Inspection statistics</div>', unsafe_allow_html=True)

high = sum(1 for h in st.session_state.history if h["severity"] == "High")
c1, c2, c3 = st.columns(3)
c1.markdown(
    f'<div class="stat"><div class="num" style="color:#1d4e89">{st.session_state.total}</div>'
    f'<div class="lbl">Items inspected</div></div>', unsafe_allow_html=True)
c2.markdown(
    f'<div class="stat"><div class="num" style="color:#dc2626">{high}</div>'
    f'<div class="lbl">High severity defects</div></div>', unsafe_allow_html=True)
c3.markdown(
    f'<div class="stat"><div class="num" style="color:#059669">{len(st.session_state.defects)}</div>'
    f'<div class="lbl">Defect types found</div></div>', unsafe_allow_html=True)

st.write("")
if st.session_state.defects:
    st.markdown('<div class="section-title">Defects by type</div>', unsafe_allow_html=True)
    mx = max(st.session_state.defects.values())
    rows = "".join(
        f'<div class="row"><div class="name">{n}</div>'
        f'<div class="track"><div class="fill" style="width:{v / mx * 100:.0f}%"></div></div>'
        f'<div class="cnt">{v}</div></div>'
        for n, v in st.session_state.defects.items()
    )
    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Inspection log</div>', unsafe_allow_html=True)
    body = "".join(
        f"<tr><td>{h['item']}</td><td>{h['defect']}</td>"
        f"<td style=\"color:{SEV_COLOR[h['severity']]} !important;font-weight:700\">{h['severity']}</td>"
        f"<td>{h['conf']}</td></tr>"
        for h in st.session_state.history[-20:][::-1]
    )
    st.markdown(
        '<table class="log"><tr><th>Item</th><th>Defect</th><th>Severity</th><th>Confidence</th></tr>'
        f"{body}</table>",
        unsafe_allow_html=True,
    )
else:
    notice("info", "No items recorded yet. Inspect an image or start a live stream.")