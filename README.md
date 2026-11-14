# RoboLabel AI — Studio Gán Nhãn & Phân Đoạn Ảnh Tự Động Cho YOLO

> **Phiên bản hiện tại:** `v1.12` — Hiển thị badge phiên bản trên Header & hỗ trợ tự động cập nhật 1-click qua `update.bat`.

Ứng dụng web chạy cục bộ giúp gán nhãn dữ liệu hình ảnh bán tự động cho **YOLO Instance Segmentation** (Polygon) và **YOLO Object Detection** (Bounding Box), tích hợp **SAM 2.1 (Segment Anything)**, **CLIP ViT-B/32** (tự động nhận dạng class), quản lý nhiều Dataset Project kèm **Auto-Save**, và công cụ **Chuyển đổi GPU/CPU + Benchmark GPU**.

---

## 1. Cấu hình Tối thiểu, Khuyến nghị & Yêu cầu phần mềm

### 1.1. Bảng cấu hình phần cứng (đối chiếu theo dung lượng thực tế & ngưỡng VRAM của hệ thống)

| Thành phần | Cấu hình Tối thiểu (Chạy được bằng CPU) | Cấu hình Khuyến nghị (Chạy mượt với GPU CUDA) |
| :--- | :--- | :--- |
| **Hệ điều hành** | Windows 10 / 11 (64-bit) | Windows 10 / 11 (64-bit) |
| **CPU** | Intel / AMD 64-bit, tối thiểu **4 luồng** (`~1.5s/vật thể` với SAM 2.1 Tiny trên CPU) | Intel Core i5/i7 (thế hệ 10+) hoặc AMD Ryzen 5/7 (**6–8 nhân / 12–16 luồng trở lên**) |
| **RAM** | **8 GB RAM** | **16 GB RAM trở lên** *(thoải mái cho ảnh độ phân giải lớn + SAM + CLIP + trình duyệt)* |
| **GPU & VRAM** | Không bắt buộc GPU rời *(hệ thống tự dùng CPU và có cơ chế tự lùi về CPU nếu GPU tràn VRAM)* | - Dùng **SAM 2.1 Tiny + CLIP**: GPU NVIDIA CUDA **≥ 4 GB VRAM** *(như GTX 1650, `~0.05s/vật thể`)*<br>- Dùng **SAM 2.1 Large (chi tiết cao)**: GPU NVIDIA CUDA **≥ 6 GB – 8 GB VRAM** *(RTX 2060 / 3060+)* |
| **Ổ cứng trống** | **8 – 10 GB trống** *(đã đo thực tế `venv` CUDA ~4.6 GB + weights mặc định ~412 MB)* | **15 – 20 GB+ SSD** *(dư cho cả SAM 2.1 Large ~428 MB, dataset nhiều dự án & Export YOLO)* |
| **Trình duyệt** | Google Chrome / Microsoft Edge bản mới *(hỗ trợ HTML5 Canvas & chọn thư mục)* | Google Chrome / Microsoft Edge bản mới |
| **Mạng Internet** | Cần khi cài đặt lần đầu (`setup.bat` tải ~2.45 GB PyTorch CUDA + ~412 MB weights + gói thư viện) & khi chạy `update.bat` | Sau khi cài xong, toàn bộ thao tác gán nhãn **chạy offline cục bộ** |

### 1.2. Tài nguyên của từng mô hình AI tích hợp

| Mô hình | Dung lượng file trọng số (`weights\`) | Ngưỡng VRAM ước tính | Ghi chú lựa chọn |
| :--- | :--- | :--- | :--- |
| **SAM 2.1 (Tiny)** — *Mặc định* | `~74.5 MB` (`sam2.1_t.pt`) | `~1.300 MB` *(cần ~1.9 GB trống để chạy dư dả)* | Tốc độ rất nhanh trên cả GPU 4 GB và CPU; khuyên dùng hằng ngày |
| **CLIP (ViT-B/32)** — *Tự nhận diện Class* | `~337.6 MB` (`clip\ViT-B-32.pt`) | `~950 MB` | Chỉ dùng khi bật công tắc **Tự nhận diện Class** |
| **SAM 2.1 (Large)** — *Tùy chọn* | `~428.4 MB` (`sam2.1_l.pt`) | `~3.600 MB` *(tự tải khi chọn lần đầu)* | Chi tiết cao hơn; với GPU 4 GB không nên bật liên tục nếu đang mở thêm CLIP/app đồ họa khác |

### 1.3. Phần mềm cần chuẩn bị trước

- **Python:** **3.10 – 3.14** ([python.org/downloads](https://www.python.org/downloads/)).
  > **Lưu ý quan trọng khi cài Python:** Ở màn hình cài đặt đầu tiên, bắt buộc tích chọn **`Add python.exe to PATH`** trước khi bấm *Install Now*.
- **Git for Windows:** Cần để tải dự án và dùng chức năng cập nhật tự động `update.bat`.
  - Cài nhanh qua PowerShell/CMD: `winget install --id Git.Git -e --source winget`
  - Hoặc tải bộ cài tại: [git-scm.com/download/win](https://git-scm.com/download/win) (các bước cài có thể giữ mặc định **Next**).
- **Driver NVIDIA (nếu máy có GPU NVIDIA):** Cài sẵn driver NVIDIA hỗ trợ CUDA `12.6` (hoặc `12.4`) để `setup.bat` tự kích hoạt PyTorch CUDA (`cu126`, dự phòng `cu124`). Nếu chưa chắc GPU mạnh hay yếu, sau khi chạy app bạn có thể bấm nút **GPU / CPU** → **Chạy Benchmark GPU** để app tự đo tốc độ thực tế và kiểm tra VRAM.

---

## 2. Hướng dẫn cài đặt chi tiết (từng bước)

### Bước 1: Tải mã nguồn về máy
Mở **Command Prompt (`cmd`)** hoặc **PowerShell** tại thư mục muốn lưu ứng dụng (ví dụ `D:\`) và chạy:

```cmd
git clone https://github.com/tocamban/robolable_ai.git
cd robolable_ai
```

> **Khuyến nghị:** Hãy tải bằng lệnh `git clone` thay vì bấm *Download ZIP* trên GitHub. Thư mục được clone bằng Git sẽ có sẵn lịch sử `.git` để dùng `update.bat` cập nhật tự động sau này. *(Nếu lỡ tải bản ZIP, xem Mục 6 để liên kết lại `.git` mà không cần cài lại từ đầu.)*

### Bước 2: Chạy cài đặt tự động 1-click (`setup.bat`)
Trong thư mục `robolable_ai`, nhấp đúp vào file **`setup.bat`**. Script sẽ tự động thực hiện toàn bộ các bước sau:

1. **Tạo môi trường ảo Python (`venv\`)** tách biệt, không ảnh hưởng các project khác trên máy.
2. **Tự nhận diện GPU NVIDIA (`nvidia-smi`):**
   - Nếu máy có GPU NVIDIA: tự động cài **PyTorch CUDA `cu126`** (~2.45 GB, có dự phòng `cu124`).
   - Nếu không có GPU NVIDIA: tự động cài **PyTorch bản CPU**.
3. **Cài đặt các thư viện cần thiết** trong `requirements.txt` (`Flask`, `Ultralytics`, `OpenCV`, `Pillow`, `CLIP`, `psutil`, `nvidia-ml-py`,...).
4. **Tự động tải trọng số AI mặc định** vào thư mục `weights\`:
   - `weights\sam2.1_t.pt` (**SAM 2.1 Tiny**, ~75 MB)
   - `weights\clip\ViT-B-32.pt` (**CLIP ViT-B/32**, ~338 MB)
   - *(Nếu về sau bạn chọn thêm **SAM 2.1 Large** trên giao diện, ứng dụng sẽ tự tải `weights\sam2.1_l.pt` ở lần dùng đầu tiên).*

Khi màn hình hiện thông báo **`CAI DAT HOAN TAT! BAN DA CO THE CHAY PROJECT!`**, nhấn phím bất kỳ để đóng cửa sổ.

---

## 3. Khởi chạy, Cập nhật & Tắt ứng dụng

| Thao tác | File thực thi | Mô tả chi tiết |
| :--- | :--- | :--- |
| **Khởi động app** | **`run_app.bat`** | Giải phóng port `5050` (nếu bị kẹt), khởi động server Flask và tự mở trình duyệt tại `http://127.0.0.1:5050`. *Giữ cửa sổ CMD này mở trong suốt lúc sử dụng để xem log AI.* |
| **Cập nhật bản mới** | **`update.bat`** | Tự tạm dừng port `5050`, kéo code mới nhất từ GitHub (tự bảo toàn dữ liệu ảnh/project/`venv`/`weights`), tự cài thêm thư viện nếu `requirements.txt` thay đổi. |
| **Tắt server** | **`stop_server.bat`** | Dừng tiến trình đang chạy trên port `5050` an toàn (hoặc bạn có thể bấm `Ctrl + C` / nút `X` trên cửa sổ terminal đang chạy `run_app.bat`). |

---

## 4. Hướng dẫn sử dụng giao diện chi tiết

Sau khi chạy `run_app.bat`, trình duyệt mở tại `http://127.0.0.1:5050`. Bạn cũng có thể bấm nút **Hướng dẫn** trên thanh tiêu đề (hoặc nhấn phím **`?`** / **`F1`**) bất cứ lúc nào để mở bảng hướng dẫn trực quan ngay trong ứng dụng.

### 4.1. Thanh Header (Phía trên)
Từ trái sang phải gồm các cụm điều khiển:
1. **Project hiện tại:** Hiển thị tên Workspace / Project đang mở (`Workspace Mặc định` hoặc tên Project bạn tạo); bấm vào để chuyển nhanh sang Project khác.
2. **Tên ảnh đang mở:** Hiển thị tên file ảnh đang chỉnh sửa trên Canvas.
3. **Chọn mô hình SAM:** Chuyển giữa **`SAM 2.1 (Tiny)`** (nhẹ, nhanh — khuyên dùng mặc định) và **`SAM 2.1 (Large)`** (viền chi tiết hơn cho vật thể phức tạp).
4. **Nút `GPU / CPU`:** Hiển thị thiết bị đang chạy (`CUDA` hoặc `CPU`). Bấm vào để:
   - Chuyển đổi linh hoạt giữa **GPU CUDA** và **CPU**.
   - Bấm **Chạy Benchmark GPU** để đo tốc độ thực tế của SAM 2.1, CLIP và kiểm tra mức VRAM khả dụng.
5. **Trạng thái Auto-Save (`Đã lưu tự động`):** Báo hiệu mọi thao tác đều được đồng bộ xuống ổ đĩa.
6. **Cụm thao tác bên phải:**
   - Nhãn phiên bản (ví dụ `v1.12`).
   - **Hướng dẫn (`?` / `F1`):** Xem sơ đồ giao diện và hình minh họa từng công cụ.
   - **Quản lý Project:** Tạo Project mới từ thư mục ảnh/dataset hoặc mở lại Project đã lưu.
   - **Export YOLO:** Xuất bộ dữ liệu hoàn chỉnh sang cấu trúc `train/val` chuẩn YOLO kèm `data.yaml`.
   - **Tải ảnh:** Thêm nhanh ảnh đơn lẻ vào Workspace/Project hiện tại.

### 4.2. Tạo & Quản lý Dataset Project (Khuyên dùng)
Thay vì gán nhãn trực tiếp lên thư mục gốc, bạn nên tạo Project riêng để bảo vệ toàn vẹn dữ liệu gốc:

1. **Cách 1 — Kéo thả trực tiếp:** Kéo cả thư mục chứa ảnh (hoặc Dataset YOLO Detection / Segmentation đã có nhãn sẵn) thả thẳng vào cửa sổ trình duyệt.
2. **Cách 2 — Qua nút `Quản lý Project`:** Chọn tab **Tạo Project Mới**, chọn thư mục nguồn từ ổ đĩa (hoặc upload).
3. Hệ thống tự phân tích loại dữ liệu (thư mục ảnh thường / YOLO Detection / YOLO Segmentation), cho phép bạn chỉnh danh sách **Classes** (hoặc chọn nhanh bộ 5 class dâu tây mặc định: `almost_ripe`, `calyx`, `stem`, `unripe`, `ripe`), đặt **Tên Project** và **Thư mục lưu riêng**.
4. Dữ liệu gốc được sao chép sang thư mục project mới, kèm file quản lý `robolabel_project.json` — lần sau có thể mở lại đúng tiến độ đang làm dở.

### 4.3. Cột trái (Classes & Current Objects) & Cột phải (Images)
- **Công tắc `Tự nhận diện Class`:**
  - **BẬT:** Khi dùng **AI Smart Box** hoặc **AI Lasso**, mô hình **CLIP ViT-B/32** sẽ tự phân tích vật thể vừa cắt để chọn tên Class phù hợp nhất.
  - **TẮT:** Gán đúng theo Class bạn đang click chọn thủ công ở danh sách **Classes** bên dưới.
- **Danh sách `Current Objects`:** Liệt kê tất cả nhãn trên ảnh hiện tại. Click chọn một đối tượng trong danh sách (hoặc click trực tiếp trên ảnh khi đang ở chế độ `V`) để hiện bảng thao tác nhanh **Tẩy bớt** / **Đắp thêm**, đổi Class hoặc xóa riêng đối tượng đó.
- **Cột `Images` bên phải:** Danh sách thumbnail toàn bộ ảnh trong Project kèm số lượng nhãn đã gắn trên mỗi ảnh để chuyển ảnh nhanh.

### 4.4. Bộ công cụ vẽ & tinh chỉnh (Thanh công cụ dọc bên phải Canvas)

#### Nhóm công cụ gán nhãn bằng AI & Khung chuẩn
- **AI Smart Box (`S`) — *Khuyên dùng*:** Giữ chuột trái kéo khung chữ nhật bao quanh vật thể → SAM 2.1 tự động tách đường viền Polygon mượt (tự khử nhiễu, không nối bắc cầu các đảo nhỏ).
- **AI Lasso:** Giữ chuột vẽ vòng bao tự do quanh đối tượng (rất hữu ích khi nhiều vật thể dính sát nhau) → SAM 2.1 chỉ tách mask nằm trong vùng bạn khoanh.
- **AI Point:** Click chọn Class ở cột trái trước, sau đó click 1 điểm vào chính giữa đối tượng để SAM tự khoanh viền.
- **Bounding Box thông thường:** Kéo khung chữ nhật YOLO Detection (4 đỉnh) không chạy AI.

#### Nhóm công cụ vẽ thủ công & sửa lỗi viền Polygon
Khi AI tách hơi lem ra ngoài hoặc thiếu một phần góc, bạn không cần xóa vẽ lại:
- **Cục Tẩy (`E`):**
  1. Chọn đối tượng cần sửa (trên ảnh hoặc trong `Current Objects`, hoặc nhấn nút **Tẩy bớt**).
  2. Chỉnh kích thước đầu tẩy (`5px – 80px`) bằng thanh trượt phía trên, các nút nhanh `10px / 20px / 35px / 50px`, hoặc phím **`[`** / **`]`**.
  3. Giữ chuột trái quẹt qua vùng bị tràn để gọt gọn đường viền.
- **Vẽ Polygon Thủ Công / Đắp thêm (`P`):**
  - **Nếu ĐANG chọn sẵn 1 đối tượng** (hiện badge tím *Đắp thêm vào Object đang chọn*): chấm các đỉnh bao lấy phần bị cắt thiếu rồi đóng kín lại → hệ thống tự động **gộp (Union)** vào khối Polygon cũ.
  - **Nếu CHƯA chọn đối tượng nào** (bấm *Bỏ chọn* nếu cần): click chấm từng đỉnh quanh vật thể để tạo Polygon mới. Di chuột lại gần điểm đầu tiên (xuất hiện vòng tròn xanh **Snap**), nhấn **`Enter`** hoặc **Double-click** để khép kín. Nhấn **Chuột phải** / **`Backspace`** / **`Delete`** để xóa đỉnh vừa đặt; nhấn **`Esc`** để hủy.
- **Chọn & Kéo đỉnh (`V`):** Click chọn đối tượng để hiển thị các chấm đỉnh trên viền, sau đó giữ chuột trái kéo từng đỉnh tới vị trí mong muốn (hệ thống tự lưu ngay khi thả chuột).
- **Di chuyển ảnh / Pan (`H` hoặc giữ chuột giữa)** & **Zoom:** Cuộn con lăn chuột để phóng to/thu nhỏ tại vị trí con trỏ, hoặc dùng cụm `- / 100% / + / RESET` ở thanh dưới cùng (có nút bật/tắt **Mask** phủ màu).
- **Hoàn tác (`Ctrl + Z`):** Khôi phục lại tối đa **30 bước** vẽ, sửa hoặc xóa gần nhất.

### 4.5. Bảng phím tắt nhanh

| Phím / Thao tác | Chức năng |
| :--- | :--- |
| **`S`** | Chọn công cụ **AI Smart Box** |
| **`P`** | Chọn **Vẽ Polygon Thủ Công** (hoặc **Đắp thêm** nếu đang chọn object) |
| **`E`** | Chọn **Cục Tẩy Polygon** |
| **`[`** / **`]`** | Giảm / Tăng kích thước Cục Tẩy (±5px) |
| **`V`** | Chọn chế độ **Chọn & Kéo đỉnh Polygon** |
| **`H`** hoặc **Giữ chuột giữa** | Chế độ **Bàn tay (Pan)** kéo di chuyển ảnh |
| **Cuộn con lăn chuột** | Phóng to / Thu nhỏ ảnh tại con trỏ |
| **`Enter`** | Hoàn tất Polygon thủ công đang vẽ (cần tối thiểu 3 đỉnh) |
| **Chuột phải** / **`Backspace`** / **`Delete`** | Xóa đỉnh vừa đặt khi đang vẽ Polygon thủ công |
| **`Ctrl + Z`** | Hoàn tác (Undo) thao tác gần nhất |
| **`?`** hoặc **`F1`** | Mở bảng **Hướng dẫn sử dụng** (`Esc` để đóng) |

### 4.6. Auto-Save & Export YOLO Dataset
- **Auto-Save:** Mọi thao tác thêm, tẩy, đắp, kéo đỉnh hoặc xóa đều tự động lưu tức thì vào file JSON chi tiết và file nhãn YOLO `.txt`.
- **Export YOLO:** Nhấn nút **Export YOLO** ở góc trên bên phải:
  - Tự giữ nguyên tập `train/val` nếu Project nhập từ một bộ YOLO đã chia sẵn; nếu là bộ ảnh mới hoàn toàn, tự động chia tỉ lệ mặc định **80% `train` / 20% `val`**.
  - Xuất đồng bộ vào thư mục `dataset/` (`images/train`, `images/val`, `labels/train`, `labels/val`) cùng file cấu hình **`data.yaml`** sẵn sàng đưa vào huấn luyện YOLO.

---

## 5. Cấu trúc thư mục dự án

```text
robolabel_app/
├── app.py                  # Server Flask backend, AI SAM 2.1 & CLIP
├── project_manager.py      # Quản lý Dataset Project, auto-detect YOLO & auto-save
├── device_manager.py       # Quản lý CPU / GPU CUDA & Benchmark phần cứng
├── static/
│   └── index.html          # Giao diện Canvas HTML5/JS + Modal Hướng dẫn trực quan
├── uploads/                # Ảnh mẫu / workspace mặc định
├── annotations/            # File JSON & TXT nhãn tương ứng workspace mặc định
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

## 6. Xử lý các lỗi thường gặp

1. **`update.bat` báo `[!] Thu muc nay khong chua lich su Git (.git)!`**
   - **Nguyên nhân:** Thư mục trên máy được tải bằng nút *Download ZIP* hoặc copy thủ công thiếu thư mục ẩn `.git`.
   - **Cách khắc phục giữ nguyên `venv/`, `weights/` và dữ liệu đã gán nhãn:** Mở `cmd` ngay trong thư mục dự án (gõ `cmd` trên thanh địa chỉ Explorer rồi Enter) và chạy lần lượt:
     ```cmd
     git init
     git checkout -B main
     git remote add origin https://github.com/tocamban/robolable_ai.git
     git fetch origin main
     git reset --mixed origin/main
     git checkout -- .
     ```
     Sau đó nhấp đúp `update.bat` sẽ hoạt động bình thường.
2. **Chạy `setup.bat` báo không tìm thấy Python:**
   - Cài lại Python 3.10–3.14 và nhớ tích **`Add python.exe to PATH`** rồi mở lại `setup.bat`.
3. **Máy có GPU NVIDIA nhưng app báo chạy `CPU`:**
   - Bấm vào nút **GPU / CPU** trên Header để chuyển sang **Chạy bằng GPU (CUDA)**; nếu hộp hướng dẫn báo *venv đang cài PyTorch bản CPU*, hãy chạy lại **`setup.bat`** (hoặc chạy lệnh cài PyTorch `cu126` hiển thị sẵn trong hộp thoại) rồi khởi động lại ứng dụng.
4. **Lỗi không mở được `http://127.0.0.1:5050` hoặc kẹt port:**
   - Chạy **`stop_server.bat`** để dọn tiến trình cũ trên port `5050`, sau đó mở lại bằng **`run_app.bat`**.

---

## 7. Đóng gói & Đẩy thay đổi lên GitHub (Dành cho người phát triển)

1. Cài đặt [Git for Windows](https://git-scm.com/download/win).
2. Khi cần đẩy mã nguồn mới lên GitHub Repository của bạn, nhấp đúp **`push_to_github.bat`**, nhập URL repository `.git` (nếu là repo mới) và xác nhận commit để push lên nhánh `main`.
