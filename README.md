# FaceAttend

FaceAttend là hệ thống chấm công khuôn mặt dùng webcam laptop. Browser gửi JPEG tới FastAPI; backend dùng SCRFD phát hiện khuôn mặt và năm landmark, ArcFace tạo embedding 512 chiều, FAISS tìm kiếm vector và MongoDB lưu dữ liệu nghiệp vụ.

## Pipeline biometric

```text
JPEG webcam
  → SCRFD ONNX: face box, score, 5 landmarks
  → quality gate
  → align 5 landmarks về ArcFace 112×112
  → ArcFace ONNX: normalized 512D embedding
  → FAISS IndexFlatIP: cosine search
  → temporal confirmation
  → MongoDB attendance event hoặc review
```

`FaceEngine` không dùng `InsightFace FaceAnalysis` hoặc model pack runtime. Nó load trực tiếp hai artifact ONNX được khai báo trong `.env`:

```env
DETECTOR_MODEL_PATH=data/models/models/buffalo_l/det_10g.onnx
DETECTOR_MODEL_ID=scrfd-10g-bnkps-v1
RECOGNIZER_MODEL_PATH=data/models/models/buffalo_l/w600k_r50.onnx
RECOGNIZER_MODEL_ID=arcface-w600k-r50-v1
```

Tên thư mục `buffalo_l` chỉ là nơi đang chứa artifact đã tải trước đó. Runtime không còn phụ thuộc package InsightFace hay load bundle này. `det_10g.onnx` là SCRFD detector và `w600k_r50.onnx` là ArcFace recognizer.

## Cấu trúc mã nguồn

```text
src/
├── faceattend/
│   ├── api/                 HTTP dependencies và router theo resource
│   ├── application/         use-case enrollment, recognition, app container
│   ├── core/                Settings và lifecycle configuration
│   ├── domain/              request/response schemas
│   ├── infrastructure/      Mongo repository và FAISS index
│   ├── services/            SCRFD/ArcFace, liveness, quality, crypto, attendance, metrics
│   └── main.py              FastAPI app, lifespan và static web
├── scripts/                 tool vận hành như threshold calibration
└── web/                     giao diện webcam thuần HTML/CSS/JS
```

Luồng dependency là `api → application → infrastructure/services`. MongoDB là source of truth; FAISS rebuild từ embedding mã hóa trong MongoDB khi service khởi động.

## Chạy bằng Docker

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Mở `http://localhost:8000` và cho phép browser dùng webcam. Container chỉ nhận JPEG qua HTTP, không truy cập camera trực tiếp.

Lần chạy đầu sau khi đổi pipeline, API sẽ kiểm tra hai file ONNX. Thiếu detector hoặc recognizer thì trả lỗi rõ ràng thay vì tự tải model không kiểm soát.

## Enrollment và recognition

P0 hiện có enrollment session nhiều mẫu, quality gate, chặn trùng embedding và lưu metadata detector/recognizer/preprocessing cho template mới. P1 có review queue, evidence mã hóa, metrics và calibration script.

Embedding trong MongoDB được mã hóa bằng Fernet. Nếu `FACEATTEND_ENCRYPTION_KEY` trống, key persistent nằm tại `data/fernet.key`; phải sao lưu key cùng MongoDB. File FAISS tại `data/faiss/templates.index` chứa raw vector, nên volume `data` phải được bảo vệ bằng quyền filesystem phù hợp.

Template cũ vẫn được giữ. Dù recognizer mới dùng chính file `w600k_r50.onnx` từ bundle cũ, nên re-enroll nhân viên trước khi dùng kết quả như một migration production đã được benchmark.

## Liveness cục bộ

Mặc định `LIVENESS_MODE=disabled`, nghĩa là hệ thống không khẳng định đã chống giả mạo. Có thể đặt `LIVENESS_MODE=onnx` và cung cấp model ONNX ở `LIVENESS_MODEL_PATH`. Model nhận tensor NCHW RGB chuẩn hóa `(pixel - 127.5) / 128`; chỉ số lớp người thật lấy từ `LIVENESS_REAL_CLASS_INDEX`.

## Hiệu chỉnh threshold

```powershell
python -m scripts.calibrate_thresholds --pairs pairs.csv --target-far 0.001
```

`pairs.csv` có hai cột `same` và `score`. Cần benchmark các threshold theo camera, ánh sáng và khoảng cách thật của site thay vì tin mặc định.

## Chạy trực tiếp bằng Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
uvicorn faceattend.main:app --app-dir src --reload
```

## Benchmark latency

Đặt ảnh JPEG/PNG chứa khuôn mặt vào một thư mục riêng, không dùng ảnh enrollment production. Script gửi request tuần tự với `dry_run=true`, nên chạy SCRFD, ArcFace, quality, liveness và FAISS nhưng không tạo attendance event, review hoặc thay đổi temporal confirmation.

```powershell
python -m scripts.benchmark_latency --images .\benchmark_images --warmup 20 --iterations 100 --out .\runs\latency.json
```

Kết quả chứa `p50`, `p95`, `p99`, mean, min/max, outcome mỗi frame và thông tin máy chạy. Latency này là HTTP end-to-end từ benchmark client đến API; file ảnh được nạp trước khi bắt đầu đo. `GET /api/metrics` cũng trả percentile server-side của tối đa 10.000 request gần nhất.
