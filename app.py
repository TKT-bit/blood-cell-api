"""
Blood Cell Detection & QNN Classification API
==============================================
Pipeline:
  1. YOLOv13 phát hiện 3 lớp tế bào: WBC, PLT, RBC
  2. Cắt ảnh từng lớp, đếm số lượng
  3. Hybrid CNN+QNN (4 hoặc 8 qubit) phân loại chi tiết WBC
  4. Trả kết quả + ảnh annotated về frontend
"""

import os
import sys
import io
import base64
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

YOLO_PATH    = MODEL_DIR / "YOLO3class.pt"
QNN4_PATH     = MODEL_DIR / "best_hybrid_4qubits.pth"
QNN8_PATH     = MODEL_DIR / "best_hybrid_8qubits.pth"
QNN_FOCAL4_PATH = MODEL_DIR / "best_hybrid_clean_done.pth"  # 4-qubit + Focal Loss
QNN_FOCAL8_PATH = MODEL_DIR / "final_hybrid_clean.pth"      # 8-qubit + Focal Loss
QNN_LIB_PYC = MODEL_DIR / "quantum_circuit_simulator.cpython-312.pyc"

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES_12 = [
    "BA", "BNE", "EO", "ERB",
    "LY", "MMY", "MO", "MY",
    "MYO", "PLT", "PMY", "SNE"
]

CLASS_FULL_NAMES = {
    "BA":  "Basophil",
    "BNE": "Band Neutrophil",
    "EO":  "Eosinophil",
    "ERB": "Erythroblast",
    "LY":  "Lymphocyte",
    "MMY": "Metamyelocyte",
    "MO":  "Monocyte",
    "MY":  "Myelocyte",
    "MYO": "Myeloid",
    "PLT": "Platelet",
    "PMY": "Promyelocyte",
    "SNE": "Segmented Neutrophil",
}

YOLO_COLORS = {
    "WBC": (255, 80,  80),   # đỏ
    "PLT": (80,  160, 255),  # xanh dương
    "RBC": (80,  210, 120),  # xanh lá
}

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

# QNN dùng PyTorch tensor → chạy được cả CPU lẫn GPU
# Trên CPU sẽ chậm hơn nhưng vẫn cho kết quả đúng
QNN_AVAILABLE = True  # luôn thử chạy, fallback trong try/except nếu lỗi
log.info(f"QNN sẽ chạy trên: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# ============================================================
# LOAD QUANTUM SIMULATOR
# ============================================================
def load_quantum_lib():
    """
    Quantum simulator được compile thành .pyc (CPython 3.12).
    Copy vào thư mục làm việc rồi import.
    """
    target_pyc = BASE_DIR / "quantum_circuit_simulator.pyc"
    if not target_pyc.exists():
        if QNN_LIB_PYC.exists():
            import shutil
            shutil.copy(QNN_LIB_PYC, target_pyc)
            log.info(f"Copied quantum lib → {target_pyc}")
        else:
            raise FileNotFoundError(
                f"Không tìm thấy quantum_circuit_simulator.pyc tại {QNN_LIB_PYC}"
            )

    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    from quantum_circuit_simulator import quantum_circuit as _qc
    log.info("✅ Quantum library loaded")
    return _qc


quantum_circuit = None  # lazy-loaded

# ============================================================
# MODEL DEFINITIONS
# ============================================================
class CNN_Feature(nn.Module):
    def __init__(self, out_features: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, out_features),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class QNN_Layer(nn.Module):
    def __init__(self, n_qubits: int = 8):
        super().__init__()
        self.n_qubits = n_qubits
        self.theta = nn.Parameter(torch.randn(n_qubits))
        self.phi   = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x):
        global quantum_circuit
        if quantum_circuit is None:
            quantum_circuit = load_quantum_lib()

        outputs = []
        x_dev = x.device

        # Thử CUDA trước, fallback CPU nếu cần
        qc_device = "cuda" if torch.cuda.is_available() else "cpu"

        theta = self.theta.to(x_dev).to(torch.cfloat)
        phi   = self.phi.to(x_dev).to(torch.cfloat)

        for i in range(x.shape[0]):
            qc = quantum_circuit(self.n_qubits, device=qc_device)
            features = x[i].to(x_dev).to(torch.cfloat)

            qc.Ry_layer(features)
            qc.Ry_layer(theta)
            qc.cx_linear_layer()
            qc.cx_linear_layer()
            qc.Rz_layer(phi)

            probs = qc.probabilities().real.squeeze()
            outputs.append(probs[: self.n_qubits])

        return torch.stack(outputs).to(x_dev)


class HybridModel(nn.Module):
    def __init__(self, n_qubits: int = 8):
        super().__init__()
        self.cnn = CNN_Feature(out_features=n_qubits)
        self.qnn = QNN_Layer(n_qubits=n_qubits)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 12)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.qnn(x)
        x = self.classifier(x)
        return x


# ============================================================
# MODEL LOADER (singleton cache)
# ============================================================
_model_cache: dict = {}
_yolo_model = None


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        if not YOLO_PATH.exists():
            raise FileNotFoundError(f"YOLO model không tìm thấy: {YOLO_PATH}")
        from ultralytics import YOLO
        _yolo_model = YOLO(str(YOLO_PATH))
        log.info("✅ YOLOv13 loaded")
    return _yolo_model


def get_qnn_model(mode: str) -> HybridModel:
    """
    mode: "4"      → 4-qubit CrossEntropy
          "8"      → 8-qubit CrossEntropy
          "focal4" → 4-qubit FocalLoss
          "focal8" → 8-qubit FocalLoss
    """
    if mode not in _model_cache:
        if mode == "4":
            path, n_qubits = QNN4_PATH, 4
        elif mode == "8":
            path, n_qubits = QNN8_PATH, 8
        elif mode == "focal4":
            path, n_qubits = QNN_FOCAL4_PATH, 4
        elif mode == "focal8":
            path, n_qubits = QNN_FOCAL8_PATH, 8
        else:
            raise ValueError(f"Chế độ không hợp lệ: {mode}")

        if not path.exists():
            raise FileNotFoundError(f"QNN model không tìm thấy: {path}")

        model = HybridModel(n_qubits=n_qubits).to(device)
        checkpoint = torch.load(path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            log.info(f"✅ QNN [{mode}] loaded (epoch={checkpoint.get('epoch','?')})")
        else:
            model.load_state_dict(checkpoint)
            log.info(f"✅ QNN [{mode}] loaded")

        model.eval()
        _model_cache[mode] = model

    return _model_cache[mode]


# ============================================================
# TRANSFORM (khớp với training)
# ============================================================
qnn_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ============================================================
# INFERENCE HELPERS
# ============================================================
def classify_wbc(crop_pil: Image.Image, model: HybridModel) -> tuple[str, float]:
    """Phân loại 1 ảnh WBC crop → (class_name, confidence)"""
    tensor = qnn_transform(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return CLASS_NAMES_12[pred.item()], conf.item()


def pil_to_b64(img: Image.Image, max_size: int = 300, quality: int = 85) -> str:
    """Resize (giữ tỉ lệ) và encode base64 JPEG"""
    ratio = min(max_size / img.width, max_size / img.height, 1.0)
    new_w = int(img.width  * ratio)
    new_h = int(img.height * ratio)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img_resized.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def draw_annotated(
    image: Image.Image,
    yolo_boxes,
    wbc_results: list[dict],
) -> Image.Image:
    """
    Vẽ bbox lên ảnh gốc:
      - WBC: màu đỏ + tên lớp QNN + conf
      - PLT: màu xanh dương
      - RBC: màu xanh lá
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Tải font đơn giản
    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font_label = ImageFont.load_default()
        font_small = font_label

    wbc_idx = 0

    for box in yolo_boxes:
        cls_id = int(box.cls[0])
        class_name = yolo_boxes[0]._meta.get("names", {}).get(cls_id, str(cls_id)) if hasattr(yolo_boxes[0], "_meta") else class_name
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = YOLO_COLORS.get(class_name, (200, 200, 200))

        # Bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        if class_name == "WBC" and wbc_idx < len(wbc_results):
            r = wbc_results[wbc_idx]
            label = f"{r['class']}  {r['confidence']*100:.0f}%"
            wbc_idx += 1
        else:
            yolo_conf = float(box.conf[0])
            label = f"{class_name} {yolo_conf*100:.0f}%"

        # Label box
        bbox_text = draw.textbbox((0, 0), label, font=font_label)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        lx = max(x1, 0)
        ly = max(y1 - th - 4, 0)
        draw.rectangle([lx, ly, lx + tw + 6, ly + th + 4], fill=color)
        draw.text((lx + 3, ly + 2), label, fill="white", font=font_label)

    return annotated


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Blood Cell Detection API",
    description="YOLOv13 + Hybrid CNN-QNN for blood cell classification",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "device":       str(device),
        "cuda":         torch.cuda.is_available(),
        "qnn_available": True,  # PyTorch-based, chạy được trên CPU
        "yolo_ready":   YOLO_PATH.exists(),
        "qnn4_ready":    QNN4_PATH.exists(),
        "qnn8_ready":    QNN8_PATH.exists(),
        "qnn_focal4_ready": QNN_FOCAL4_PATH.exists(),
        "qnn_focal8_ready": QNN_FOCAL8_PATH.exists(),
        "quantum_lib":   QNN_LIB_PYC.exists(),
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    qubit_mode: str = Form("4"),
):
    """
    Endpoint chính:
      - Nhận ảnh + qubit_mode ("4", "8", "focal4")
      - Trả về JSON với ảnh crop mẫu, số lượng, và WBC detections
    """
    if qubit_mode not in ("4", "8", "focal4", "focal8"):
        raise HTTPException(400, "qubit_mode phải là '4', '8', 'focal4' hoặc 'focal8'")

    # ── Đọc ảnh ──────────────────────────────────────────
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Không đọc được ảnh: {e}")

    log.info(f"Input image: {image.size}, qubit_mode={qubit_mode}")

    # ── YOLO detection ────────────────────────────────────
    yolo = get_yolo()
    results = yolo.predict(
        source=image,
        conf=0.25,
        iou=0.45,
        save=False,
        verbose=False
    )
    result = results[0]

    # ── Load QNN model (CPU hoặc GPU đều được) ───────────────
    counts    = {"WBC": 0, "PLT": 0, "RBC": 0}
    samples   = {"WBC": None, "PLT": None, "RBC": None}
    wbc_boxes = []
    wbc_results: list[dict] = []

    qnn_model = None
    try:
        qnn_model = get_qnn_model(qubit_mode)
    except Exception as e:
        log.warning(f"Không load được QNN model: {e}")

    for box in result.boxes:
        cls_id     = int(box.cls[0])
        class_name = result.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Đảm bảo bbox hợp lệ
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.width), min(y2, image.height)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))

        if class_name in counts:
            counts[class_name] += 1
            if samples[class_name] is None:
                samples[class_name] = crop

        if class_name == "WBC":
            wbc_boxes.append(box)
            if qnn_model is not None:
                try:
                    pred_class, conf = classify_wbc(crop, qnn_model)
                    wbc_results.append({
                        "id":         counts["WBC"],
                        "class":      pred_class,
                        "full_name":  CLASS_FULL_NAMES.get(pred_class, pred_class),
                        "confidence": round(conf, 4),
                        "bbox":       [x1, y1, x2, y2],
                        "qnn_ok":     True,
                    })
                    log.info(f"  WBC #{counts['WBC']} → {pred_class} ({conf*100:.1f}%)")
                except Exception as e:
                    log.warning(f"QNN classify failed WBC #{counts['WBC']}: {e}")
                    wbc_results.append({
                        "id":         counts["WBC"],
                        "class":      "ERR",
                        "full_name":  f"Lỗi: {e}",
                        "confidence": 0.0,
                        "bbox":       [x1, y1, x2, y2],
                        "qnn_ok":     False,
                    })
            else:
                wbc_results.append({
                    "id":         counts["WBC"],
                    "class":      "N/A",
                    "full_name":  "Không load được QNN model",
                    "confidence": 0.0,
                    "bbox":       [x1, y1, x2, y2],
                    "qnn_ok":     False,
                })

    # ── Annotated image (vẽ bbox + nhãn QNN lên ảnh gốc) ──
    annotated_pil = image.copy()
    draw = ImageDraw.Draw(annotated_pil)
    try:
        font_lbl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font_lbl = ImageFont.load_default()

    wbc_idx = 0
    for box in result.boxes:
        cls_id     = int(box.cls[0])
        class_name = result.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.width), min(y2, image.height)
        if x2 <= x1 or y2 <= y1:
            continue

        color = YOLO_COLORS.get(class_name, (180, 180, 180))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        if class_name == "WBC" and wbc_idx < len(wbc_results):
            r = wbc_results[wbc_idx]
            label = f"{r['class']}  {r['confidence']*100:.0f}%"
            wbc_idx += 1
        else:
            yolo_conf = float(box.conf[0])
            label = f"{class_name} {yolo_conf*100:.0f}%"

        try:
            bbox_text = draw.textbbox((0, 0), label, font=font_lbl)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
        except Exception:
            tw, th = len(label) * 8, 16
        lx = max(x1, 0)
        ly = max(y1 - th - 4, 0)
        draw.rectangle([lx, ly, lx + tw + 8, ly + th + 4], fill=color)
        draw.text((lx + 4, ly + 2), label, fill="white", font=font_lbl)

    # ── Tạo ảnh crop WBC có nhãn phân loại ──────────────────
    wbc_annotated_crops = []

    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font_big = ImageFont.load_default()
        font_med = font_big

    for r in wbc_results:
        try:
            x1, y1, x2, y2 = r["bbox"]
            pad = 14
            cx1 = max(x1 - pad, 0)
            cy1 = max(y1 - pad, 0)
            cx2 = min(x2 + pad, image.width)
            cy2 = min(y2 + pad, image.height)

            crop = image.crop((cx1, cy1, cx2, cy2)).convert("RGB").copy()
            draw_c = ImageDraw.Draw(crop)

            # bbox bên trong crop (offset padding)
            bx1 = x1 - cx1
            by1 = y1 - cy1
            bx2 = bx1 + (x2 - x1)
            by2 = by1 + (y2 - y1)
            draw_c.rectangle([bx1, by1, bx2, by2], outline=(220, 50, 50), width=2)

            # Nội dung label
            if r.get("qnn_ok"):
                conf_pct = int(r["confidence"] * 100)
                lines = [r["class"], r["full_name"], f"Conf: {conf_pct}%"]
                bg_color = (210, 40, 40)
            else:
                lines = ["WBC", "N/A"]
                bg_color = (100, 100, 100)

            # Tính kích thước label box
            line_heights = []
            line_widths  = []
            for ln in lines:
                try:
                    bb = draw_c.textbbox((0, 0), ln, font=(font_big if ln == lines[0] else font_med))
                    line_heights.append(bb[3] - bb[1])
                    line_widths.append(bb[2] - bb[0])
                except Exception:
                    line_heights.append(16)
                    line_widths.append(80)

            lw = max(line_widths) + 12
            lh = sum(line_heights) + 4 * len(lines) + 4

            # Đặt label dưới bbox, nếu không đủ chỗ thì đặt trên
            lx = max(bx1, 0)
            ly = by2 + 4
            if ly + lh > crop.height:
                ly = max(by1 - lh - 4, 0)

            # Vẽ background label
            draw_c.rectangle([lx, ly, lx + lw, ly + lh], fill=bg_color)

            # Vẽ text từng dòng
            ty = ly + 3
            for idx, ln in enumerate(lines):
                fnt = font_big if idx == 0 else font_med
                draw_c.text((lx + 6, ty), ln, fill="white", font=fnt)
                ty += line_heights[idx] + 4

            wbc_annotated_crops.append(pil_to_b64(crop, max_size=280))
            log.info(f"  Crop WBC #{r['id']} OK ({crop.size})")

        except Exception as e:
            log.warning(f"  Crop WBC #{r.get('id','?')} lỗi: {e}")
            # Vẫn append crop gốc không có label thay vì bỏ qua
            try:
                x1, y1, x2, y2 = r["bbox"]
                fallback = image.crop((max(x1-14,0), max(y1-14,0),
                                       min(x2+14,image.width), min(y2+14,image.height)))
                wbc_annotated_crops.append(pil_to_b64(fallback, max_size=280))
            except Exception:
                pass

    # ── Encode kết quả sang base64 ─────────────────────────
    sample_b64 = {}
    for cls_name, img in samples.items():
        if img is not None:
            sample_b64[cls_name] = pil_to_b64(img, max_size=260)
        else:
            sample_b64[cls_name] = None

    return JSONResponse({
        "success":              True,
        "qubit_mode":           qubit_mode,
        "qnn_available":        qnn_model is not None,
        "counts":               counts,
        "sample_images":        sample_b64,
        "wbc_annotated_crops":  wbc_annotated_crops,
        "wbc_detections":       wbc_results,
        "total_cells":          sum(counts.values()),
    })


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )