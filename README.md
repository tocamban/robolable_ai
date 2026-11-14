# RoboLabel AI — Studio Gán Nhãn & Phân Đoạn Ảnh Tự Động Cho YOLO

Ứng dụng web chạy cục bộ giúp gán nhãn dữ liệu hình ảnh bán tự động cho **YOLO Instance Segmentation** (Polygon) và **YOLO Object Detection** (Bounding Box), tích hợp **SAM 2.1 (Segment Anything)**, **CLIP ViT-B/32** (tự động nhận dạng class), quản lý nhiều Dataset Project kèm **Auto-Save**, và công cụ **Chuyển đổi GPU/CPU + Benchmark GPU**.

> **Phiên bản hiện tại:** `v1.1` — Hiển thị badge phiên bản trên Header & hỗ trợ tự động cập nhật 1-click qua `update.bat`.

---

## 1. Tính năng chính

1. **Quản lý Dataset Project & Auto-Save:**
   - Kéo thả trực tiếp thư mục ảnh thường hoặc dataset YOLO (Detection / Segmentation) vào giao diện để tạo project mới tại thư mục tùy chọn (không ghi đè dữ liệu gốc).
   - Tự động lưu tức thì (Auto-Save) mọi thao tác thêm, sửa, xóa nhãn xuống JSON và file YOLO `.txt`.
   - Mở lại project đã lưu bất cứ lúc nào và tiếp tục từ ảnh đang làm dở.
2. **Đánh nhãn Polygon bằng AI (SAM 2.1 Tiny / Large + CLIP):**
   - **AI Smart Box (`S`):** Kéo khung bao quanh vật thể để SAM 2.1 tự tách đường viền Polygon mượt (tự khử nhiễu, không nối bắc cầu các đảo nhỏ) và CLIP tự phân loại class.
   - **AI Lasso:** Khoanh vùng tự do giới hạn mask AI bên trong vùng chọn.
   - **AI Point:** Click 1 điểm lên vật thể để tạo mặt nạ tức thì.
3. **Tinh chỉnh Polygon linh hoạt:**
   - **Cục Tẩy (`E`, `[` / `]`):** Tẩy xóa trực tiếp vùng bị lem/tràn.
   - **Đắp thêm / Polygon Thủ Công (`P`):** Vẽ bổ sung để gộp (Union) vào đối tượng đang chọn hoặc tạo mới với vòng Snap điểm đầu.
   - **Chọn & Kéo đỉnh (`V`):** Nắn chỉnh từng đỉnh đa giác trực tiếp trên Canvas.
   - **Undo (`Ctrl + Z`):** Hoàn tác nhiều bước.
4. **Chuyển đổi GPU / CPU & Benchmark GPU:**
   - Hỗ trợ chuyển đổi linh hoạt giữa GPU CUDA và CPU ngay trên thanh Header, kèm công cụ đo tốc độ thực của SAM 2.1 / CLIP và kiểm tra mức tương thích VRAM.
5. **Hướng dẫn sử dụng trực quan ngay trên UI:**
   - Bấm nút **Hướng dẫn** trên Header (hoặc phím `?` / `F1`) để xem sơ đồ giao diện và minh họa thao tác từng công cụ.
6. **Export YOLO Dataset:**
   - Tự động chia tập `train` / `val`, xuất cấu trúc thư mục ảnh/nhãn chuẩn cùng `data.yaml`.

---

## 2. Cấu trúc thư mục dự án

```text
robolabel_app/
├── app.py                  # Server Flask backend, AI SAM 2.1 & CLIP
├── project_manager.py      # Quản lý Dataset Project, auto-detect YOLO & auto-save
├── device_manager.py       # Quản lý CPU / GPU CUDA & Benchmark phần cứng
├── static/
│   └── index.html          # Giao diện Canvas HTML5/JS + Modal Hướng dẫn trực quan
├── uploads/                # Ảnh mẫu / workspace mặc định
├── annotations/            # File JSON nhãn tương ứng
├── weights/                # Thư mục lưu trọng số AI (SAM 2.1 & CLIP - tự tải bởi setup.bat)
├── requirements.txt        # Danh sách thư viện Python
├── setup.bat               # Tự động tạo venv, cài CUDA/CPU PyTorch & tải model AI
├── run_app.bat             # Khởi động server (port 5050) và mở trình duyệt
├── update.bat              # 1-click kéo bản cập nhật mới nhất từ GitHub
├── stop_server.bat         # Tắt server & giải phóng port 5050
├── stop_server.ps1         # Script PowerShell dừng tiến trình trên port 5050
└── push_to_github.bat      # Hỗ trợ đóng gói & đẩy mã nguồn lên GitHub
```

*(Các thư mục nặng như `venv/` và file trọng số `weights/*.pt` được loại trừ trong `.gitignore` để giữ repository nhẹ và không vượt quá giới hạn 100 MB của GitHub.)*

---

## 3. Cài đặt & Chạy sau khi clone từ GitHub

Yêu cầu: Windows, **Python 3.10 – 3.14** (nhớ tích chọn **Add python.exe to PATH** khi cài).

1. **Clone mã nguồn về máy:**
   ```bash
   git clone https://github.com/tocamban/robolable_ai.git
   cd robolable_ai
   ```
2. **Cài đặt tự động (1-click):**
   - Nhấp đúp vào **`setup.bat`**. Script sẽ tự tạo `venv`, tự nhận diện GPU NVIDIA để cài PyTorch CUDA (`cu126`), cài các gói trong `requirements.txt`, và tự tải model **SAM 2.1 Tiny** + **CLIP ViT-B/32** vào thư mục `weights\`.
   - *(Nếu dùng thêm model `SAM 2.1 Large`, app sẽ tự động tải về `weights\sam2.1_l.pt` ở lần chọn đầu tiên).*
3. **Khởi chạy ứng dụng:**
   - Nhấp đúp **`run_app.bat`** → trình duyệt mở tại `http://127.0.0.1:5050`.
4. **Cập nhật phiên bản mới (trên máy khác):**
   - Mỗi khi có bản cập nhật mới trên GitHub, chỉ cần nhấp đúp **`update.bat`** (tự động pull code mới và cài thêm thư viện nếu có thay đổi).
5. **Tắt ứng dụng:**
   - Nhấp đúp **`stop_server.bat`** hoặc bấm `Ctrl + C` trong cửa sổ terminal.

---

## 4. Đóng gói & Đẩy dự án lên GitHub

1. Cài đặt [Git for Windows](https://git-scm.com/download/win) (hoặc chạy `winget install --id Git.Git -e --source winget`).
2. Tạo một Repository mới (trống, chưa khởi tạo README) trên [github.com/new](https://github.com/new).
3. Nhấp đúp **`push_to_github.bat`** trong thư mục dự án, nhập URL repository `.git` khi được hỏi và đăng nhập GitHub để hoàn tất.
