"""Pure helper functions for SteelScan AI.

No Streamlit code lives here, so everything can be tested on its own.
"""
import csv
import datetime
import functools
import io
import textwrap
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CONF_THRESHOLD = 0.6

# severity -> (verdict text, colour)
_VERDICTS = {
    "High": ("REJECT", "#dc2626"),
    "Medium": ("REWORK", "#d97706"),
    "Low": ("PASS WITH NOTE", "#059669"),
}


# ------------------------------------------------------------------ verdict
def verdict_for(severity, conf, threshold=CONF_THRESHOLD):
    """Turn severity + confidence into a factory decision."""
    if conf < threshold:
        return "MANUAL CHECK", "#64748b"
    return _VERDICTS.get(severity, ("MANUAL CHECK", "#64748b"))


# -------------------------------------------------------------------- sound
@functools.lru_cache(maxsize=128)
def beep_wav_bytes(variant=0):
    """Two short beeps as WAV bytes.

    `variant` adds a few silent samples so the bytes differ each time; the
    browser then treats it as a new audio file and plays it again.
    """
    sr = 22050

    def tone(sec, freq=880.0):
        t = np.linspace(0, sec, int(sr * sec), endpoint=False)
        return 0.5 * np.sin(2 * np.pi * freq * t)

    gap = np.zeros(int(sr * 0.1))
    tail = np.zeros(int(variant) % 200 + 1)
    data = np.concatenate([tone(0.2), gap, tone(0.2), tail])
    pcm = (data * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------- ROI
def inr(n):
    """Format a number in Indian rupees with lakh/crore grouping."""
    n = int(round(max(n, 0)))
    s = str(n)
    if len(s) <= 3:
        return "\u20b9" + s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return "\u20b9" + ",".join(parts + [tail])


def roi_estimate(items_per_hour, manual_sec, wage, hours_day, days_month,
                 automation_pct, defect_rate_pct, manual_miss_pct,
                 ai_miss_pct, cost_per_escape):
    """Rough monthly savings from automating inspection (all inputs are assumptions)."""
    inspectors = items_per_hour * manual_sec / 3600.0
    hours_saved = inspectors * hours_day * days_month * automation_pct / 100.0
    labour = hours_saved * wage

    defects_month = items_per_hour * hours_day * days_month * defect_rate_pct / 100.0
    escaped_manual = defects_month * manual_miss_pct / 100.0
    escaped_ai = defects_month * ai_miss_pct / 100.0
    fewer_escapes = max(escaped_manual - escaped_ai, 0.0)
    quality = fewer_escapes * cost_per_escape

    monthly = labour + quality
    return {
        "inspectors": inspectors,
        "hours_saved": hours_saved,
        "labour": labour,
        "fewer_escapes": fewer_escapes,
        "quality": quality,
        "monthly": monthly,
        "yearly": monthly * 12,
    }


# ------------------------------------------------------------------ Grad-CAM
def gradcam(model, pil_img):
    """Grad-CAM for an Ultralytics YOLOv8 *classification* model.

    Returns (cam, error). `cam` is a 2-D float array (unnormalised) or None.
    Never raises, so the app keeps working even if this fails.
    """
    try:
        import torch

        net = model.model
        net.eval()
        layers = net.model
        target = layers[-2]          # last feature block before Classify head
        head = layers[-1]
        store = {}

        def fwd_hook(_m, _i, out):
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))

        def logit_hook(_m, _i, out):
            store["logits"] = out

        handles = [target.register_forward_hook(fwd_hook)]
        if hasattr(head, "linear"):
            handles.append(head.linear.register_forward_hook(logit_hook))

        try:
            param = next(net.parameters())
            gray = pil_img.convert("L").convert("RGB").resize((224, 224))
            x = torch.from_numpy(np.asarray(gray).copy()).permute(2, 0, 1).unsqueeze(0)
            x = (x.to(param.device).to(param.dtype) / 255.0)
            x.requires_grad_(True)

            with torch.enable_grad():
                out = net(x)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                idx = int(out[0].argmax())
                score = store.get("logits", out)[0, idx]
                net.zero_grad()
                score.backward()
        finally:
            for h in handles:
                h.remove()

        act = store["act"].detach()[0]          # C x H x W
        grad = store["grad"].detach()[0]
        weights = grad.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * act).sum(0)).cpu().numpy()
        return cam, None
    except Exception as e:  # noqa: BLE001 - we never want this to crash the app
        return None, f"{type(e).__name__}: {e}"


def overlay_cam(pil_img, cam, region_threshold=0.6):
    """Return (heatmap overlay, image with defect-region box) as RGB arrays."""
    img = np.asarray(pil_img.convert("RGB")).copy()
    h, w = img.shape[:2]
    cam = np.asarray(cam, dtype=np.float32)
    cam = cam - cam.min()
    peak = cam.max()
    cam = cam / peak if peak > 1e-8 else np.zeros_like(cam)
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0, 1)

    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img, 0.55, heat, 0.45, 0)

    boxed = img.copy()
    mask = (cam >= region_threshold).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        thick = max(2, int(min(h, w) / 100))
        cv2.rectangle(boxed, (x, y), (x + bw, y + bh), (220, 38, 38), thick)
    return blended, boxed


# ---------------------------------------------------------------------- CSV
def history_to_csv(history):
    """Inspection log -> CSV bytes (opens fine in Excel)."""
    cols = ["item", "time", "defect", "severity", "verdict", "conf"]
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    wr.writeheader()
    for row in history:
        wr.writerow(row)
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


# ---------------------------------------------------------------------- PDF
def _font(size, bold=False):
    names = (["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"] if bold
             else ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:  # noqa: BLE001
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    lines = []
    for para in str(text).split("\n"):
        words, cur = para.split(), ""
        for w in words:
            trial = (cur + " " + w).strip()
            if draw.textlength(trial, font=font) <= max_w:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def _fit(arr_or_img, size):
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    img = img.convert("RGB")
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (241, 245, 249))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def build_pdf_report(img, heat, box, label, severity, verdict, verdict_color,
                     conf, desc, action, top3, ai_text=None, when=None):
    """One-page A4 inspection report as PDF bytes (Pillow only, no extra library)."""
    W, H, M = 1240, 1754, 70
    when = when or datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    page = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(page)
    dark, text, muted = (11, 29, 51), (30, 41, 59), (100, 116, 139)

    # header
    d.rectangle([0, 0, W, 150], fill=dark)
    d.text((M, 38), "SteelScan AI", font=_font(54, True), fill="white")
    d.text((M, 100), "Steel surface inspection report", font=_font(26), fill=(203, 213, 225))
    d.text((W - M - 330, 100), when, font=_font(24), fill=(203, 213, 225))

    # verdict badge
    y = 190
    vtxt = f"VERDICT: {verdict}"
    vf = _font(40, True)
    tw = d.textlength(vtxt, font=vf)
    d.rounded_rectangle([M, y, M + tw + 60, y + 80], radius=12, fill=verdict_color)
    d.text((M + 30, y + 16), vtxt, font=vf, fill="white")

    # key facts
    y = 310
    for k, v in [("Defect", label), ("Severity", severity), ("Model confidence", f"{conf:.0%}")]:
        d.text((M, y), k, font=_font(26), fill=muted)
        d.text((M + 280, y), str(v), font=_font(30, True), fill=text)
        y += 52

    # images
    y = 520
    size, gap = 340, 50
    shots = [(img, "Inspected image"), (heat, "Grad-CAM heatmap"), (box, "Approx. defect region")]
    x = M
    for im, cap in shots:
        if im is not None:
            page.paste(_fit(im, size), (x, y))
        else:
            d.rectangle([x, y, x + size, y + size], outline=(203, 213, 225), width=2)
            d.text((x + 70, y + size // 2 - 12), "not available", font=_font(24), fill=muted)
        d.text((x, y + size + 12), cap, font=_font(22), fill=muted)
        x += size + gap

    # top-3
    y = 940
    d.text((M, y), "Top predictions", font=_font(28, True), fill=text)
    y += 50
    for name, c in top3[:3]:
        d.text((M, y), name, font=_font(24), fill=text)
        d.rectangle([M + 320, y + 4, M + 320 + 600, y + 28], fill=(226, 232, 240))
        d.rectangle([M + 320, y + 4, M + 320 + int(600 * c), y + 28], fill=(29, 78, 137))
        d.text((M + 940, y), f"{c:.0%}", font=_font(24, True), fill=text)
        y += 44

    # description / action
    y += 20
    bf = _font(25)
    for head, body in [("Description", desc), ("Recommended action", action)]:
        d.text((M, y), head, font=_font(28, True), fill=text)
        y += 44
        for line in _wrap(d, body, bf, W - 2 * M):
            d.text((M, y), line, font=bf, fill=text)
            y += 36
        y += 18

    # AI report
    if ai_text:
        d.text((M, y), "AI inspection report", font=_font(28, True), fill=text)
        y += 44
        clean = ai_text.replace("**", "")
        for line in _wrap(d, clean, bf, W - 2 * M):
            if y > H - 140:
                break
            d.text((M, y), line, font=bf, fill=text)
            y += 36

    # footer
    d.line([M, H - 100, W - M, H - 100], fill=(203, 213, 225), width=2)
    d.text((M, H - 80), "Generated by SteelScan AI  |  Team Zenith  |  AI Manthan 2.0",
           font=_font(22), fill=muted)

    buf = io.BytesIO()
    page.save(buf, format="PDF", resolution=150.0)
    return buf.getvalue()