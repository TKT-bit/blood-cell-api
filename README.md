# Blood Cell Analysis — Web API

**YOLOv13 detection + Hybrid CNN–QNN classification**

---

## Cấu trúc project

```
blood_cell_api/
├── app.py                 # FastAPI backend
├── requirements.txt
├── static/
│   └── index.html         # Frontend (light theme)
├── models/                # ← ĐẶT CÁC FILE MODEL VÀO ĐÂY
│   ├── YOLO3class.pt
│   ├── best_hybrid_4qubits.pth
│   ├── best_hybrid_8qubits.pth
│   ├──
│   ├── 
│   └── quantum_circuit_simulator.cpython.pyc
└── README.md
```

---

## Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| Python | **3.12** (do quantum_circuit_simulator.pyc được compile cho CPython 3.12) |
| CUDA (khuyến nghị) | CUDA 12.x + GPU NVIDIA |
| RAM | ≥ 8 GB |

> ⚠️ **Lưu ý quan trọng**: Quantum simulator trong `QNN_Layer` gọi `quantum_circuit(device='cuda')`. Cần có CUDA để chạy QNN inference. Nếu chỉ có CPU, bước QNN sẽ báo lỗi — xem phần "Chạy CPU-only" bên dưới.

---

## Cài đặt

```bash
# 1. Tạo virtualenv Python 3.12
python3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Sao chép model files vào thư mục models/
cp /path/to/YOLO3class.pt                              models/
cp /path/to/best_hybrid_4qubits.pth                   models/
cp /path/to/best_hybrid_8qubits.pth                   models/
cp /path/to/quantum_circuit_simulator.cpython-312.pyc models/
```

---

## Chạy server

```bash
python app.py
```

Hoặc với uvicorn trực tiếp:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Mở trình duyệt: **http://localhost:8000**

---

## API Endpoints

### `GET /`
Trả về trang web frontend.

### `GET /health`
Kiểm tra trạng thái server và các model files.

```json
{
  "status": "ok",
  "device": "cuda",
  "cuda": true,
  "yolo_ready": true,
  "qnn4_ready": true,
  "qnn8_ready": true,
  "quantum_lib": true
}
```

### `POST /api/analyze`

| Field | Type | Mô tả |
|---|---|---|
| `file` | `File` | Ảnh tiêu bản máu (JPG/PNG/WEBP) |
| `qubit_mode` | `int` | `4` hoặc `8` |

**Response JSON:**

```json
{
  "success": true,
  "qubit_mode": 4,
  "counts": { "WBC": 3, "PLT": 12, "RBC": 45 },
  "total_cells": 60,
  "sample_images": {
    "WBC": "<base64 JPEG>",
    "PLT": "<base64 JPEG>",
    "RBC": "<base64 JPEG>"
  },
  "annotated_image": "<base64 JPEG>",
  "wbc_detections": [
    {
      "id": 1,
      "class": "LY",
      "full_name": "Lymphocyte",
      "confidence": 0.9231,
      "bbox": [120, 80, 210, 170]
    }
  ]
}
```

---

## 12 lớp WBC

| Code | Tên đầy đủ |
|---|---|
| BA  | Basophil |
| BNE | Band Neutrophil |
| EO  | Eosinophil |
| ERB | Erythroblast |
| LY  | Lymphocyte |
| MMY | Metamyelocyte |
| MO  | Monocyte |
| MY  | Myelocyte |
| MYO | Myeloid |
| PLT | Platelet |
| PMY | Promyelocyte |
| SNE | Segmented Neutrophil |

---

## Pipeline xử lý

```
Ảnh đầu vào
    │
    ▼
YOLOv13 (YOLO3class.pt)
    │  detect: WBC / PLT / RBC
    ├──────────────────────────────┐
    │                              │
    ▼                              ▼
Cắt bbox từng class           Đếm số lượng
Lấy 1 ảnh mẫu/class          WBC / PLT / RBC
    │
    ▼ (chỉ WBC)
Hybrid CNN–QNN
(4 hoặc 8 qubit)
    │
    ▼
Phân loại 12 lớp WBC
+ Confidence score
    │
    ▼
Vẽ bbox + nhãn lên ảnh gốc
    │
    ▼
JSON Response → Frontend
```

---

## Chạy CPU-only (không có GPU)

Quantum simulator được hardcode `device='cuda'`. Để chạy trên CPU, cần patch trong `QNN_Layer.forward()`:

```python
# Trong app.py, sửa dòng này:
qc = quantum_circuit(self.n_qubits, device=qc_device)

# qc_device hiện tại là:
qc_device = "cuda" if torch.cuda.is_available() else "cpu"
```

Nếu `quantum_circuit` không hỗ trợ `device='cpu'`, cần có phiên bản `.pyc` compile phù hợp hoặc thư viện nguồn.

---

## Lưu ý triển khai

- Model được cache sau lần load đầu tiên — inference lần 2 trở đi nhanh hơn đáng kể.
- Ảnh crop mẫu được resize về tối đa 260px; ảnh annotated tối đa 700px.
- YOLO conf threshold: 0.25, IoU: 0.45 (có thể điều chỉnh trong `app.py`).
