"""Evaluate the SteelScan model and write results into ./metrics/

Usage:
    python evaluate.py --data path/to/val            # val/ has one sub-folder per class
    python evaluate.py --data path/to/val --skip-onnx

Creates:
    metrics/summary.json          accuracy, per-class precision/recall/F1, robustness, latency
    metrics/confusion_matrix.png
    metrics/robustness.png
The Streamlit app shows these automatically in the "Model performance" section.
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ------------------------------------------------------------------ data
def load_dataset(root, limit=None):
    """root/<class_name>/*.jpg -> (list of BGR images, list of class names)."""
    images, labels = [], []
    for cls in sorted(os.listdir(root)):
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                       if f.lower().endswith(IMG_EXT))
        if limit:
            files = files[:limit]
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                images.append(img)
                labels.append(cls)
    return images, labels


# ------------------------------------------------------------ perturbations
def _noise(sigma):
    def f(img):
        rng = np.random.default_rng(0)
        out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
        return np.clip(out, 0, 255).astype(np.uint8)
    return f


def _jpeg(q):
    def f(img):
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return f


PERTURBATIONS = {
    "clean": lambda im: im,
    "blur (5x5)": lambda im: cv2.GaussianBlur(im, (5, 5), 0),
    "blur (9x9)": lambda im: cv2.GaussianBlur(im, (9, 9), 0),
    "darker (x0.6)": lambda im: np.clip(im.astype(np.float32) * 0.6, 0, 255).astype(np.uint8),
    "brighter (x1.4)": lambda im: np.clip(im.astype(np.float32) * 1.4, 0, 255).astype(np.uint8),
    "noise (sigma 15)": _noise(15),
    "noise (sigma 30)": _noise(30),
    "JPEG quality 30": _jpeg(30),
}


# ----------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, class_names):
    idx = {c: i for i, c in enumerate(class_names)}
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    per_class = {}
    for c, i in idx.items():
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": float(prec), "recall": float(rec),
                        "f1": float(f1), "support": int(cm[i, :].sum())}
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    return acc, per_class, cm


def plot_confusion(cm, class_names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    top = cm.max() / 2 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > top else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_robustness(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names, vals = list(results), [results[k] * 100 for k in results]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, vals, color=["#1d4e89"] + ["#d97706"] * (len(names) - 1))
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy under image degradation")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------- inference
def to_gray3(img):
    """Same preprocessing as the app: grayscale, then back to 3 channels."""
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def predict_labels(model, images, batch=32):
    out = []
    for i in range(0, len(images), batch):
        chunk = [to_gray3(im) for im in images[i:i + batch]]
        results = model.predict(chunk, imgsz=224, verbose=False, device="cpu")
        out.extend(model.names[r.probs.top1] for r in results)
    return out


def evaluate_model(model, root, out_dir, limit=None):
    images, labels = load_dataset(root, limit)
    if not images:
        raise SystemExit(f"No images found under {root}. Expected {root}/<class_name>/*.jpg")
    class_names = [model.names[i] for i in sorted(model.names)]
    unknown = sorted(set(labels) - set(class_names))
    if unknown:
        print(f"Warning: folders not in model classes were skipped: {unknown}")
        keep = [i for i, l in enumerate(labels) if l in class_names]
        images, labels = [images[i] for i in keep], [labels[i] for i in keep]

    os.makedirs(out_dir, exist_ok=True)
    print(f"Evaluating {len(images)} images ...")
    pred = predict_labels(model, images)
    acc, per_class, cm = compute_metrics(labels, pred, class_names)
    plot_confusion(cm, class_names, os.path.join(out_dir, "confusion_matrix.png"))
    print(f"Accuracy: {acc:.2%}")

    robustness = {}
    for name, fn in PERTURBATIONS.items():
        if name == "clean":
            robustness[name] = acc
            continue
        p = predict_labels(model, [fn(im) for im in images])
        robustness[name] = float(np.mean([a == b for a, b in zip(labels, p)]))
        print(f"  {name:<18} {robustness[name]:.2%}")
    plot_robustness(robustness, os.path.join(out_dir, "robustness.png"))
    return {"accuracy": acc, "n_images": len(images), "per_class": per_class,
            "robustness": robustness}, images


def time_model(model, sample, runs=50):
    img = to_gray3(sample)
    for _ in range(5):
        model.predict(img, imgsz=224, verbose=False, device="cpu")
    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict(img, imgsz=224, verbose=False, device="cpu")
    return (time.perf_counter() - t0) / runs * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--out", default="metrics")
    ap.add_argument("--limit", type=int, default=None, help="max images per class")
    ap.add_argument("--skip-onnx", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    summary, images = evaluate_model(model, args.data, args.out, args.limit)

    latency = {"pytorch_cpu": time_model(model, images[0])}
    if not args.skip_onnx:
        try:
            onnx_path = model.export(format="onnx", imgsz=224)
            onnx_model = YOLO(onnx_path, task="classify")
            latency["onnx_cpu"] = time_model(onnx_model, images[0])
        except Exception as e:  # noqa: BLE001
            print(f"ONNX step skipped: {e}")
    summary["latency_ms"] = latency
    print("Latency (ms/image, CPU):", {k: round(v, 1) for k, v in latency.items()})

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results in ./{args.out}/")


if __name__ == "__main__":
    main()