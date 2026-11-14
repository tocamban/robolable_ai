import gc
import os
import sys
import json
import shutil
import random
import subprocess
import threading
from urllib.parse import quote
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from ultralytics import SAM

from device_manager import (
    get_hardware_info,
    load_device_preference,
    resolve_runtime_device,
    run_device_benchmark,
    save_device_preference,
)
from project_manager import (
    DEFAULT_CLASS_INFO,
    DEFAULT_CLASS_NAMES,
    ProjectManager,
    inspect_dataset_folder,
    is_image_file,
)

app = Flask(__name__, static_folder="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ANNOTATION_DIR = os.path.join(BASE_DIR, "annotations")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ANNOTATION_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

pm = ProjectManager(BASE_DIR)

# Hardware & Device detection (respects user's saved CPU/GPU preference)
DEVICE_LOCK = threading.Lock()
PREFERRED_DEVICE = load_device_preference(BASE_DIR)
DEVICE, DEVICE_NAME, DEVICE_FALLBACK_NOTE = resolve_runtime_device(PREFERRED_DEVICE)
print(f"[*] Running on device: {DEVICE} ({DEVICE_NAME}) [pref={PREFERRED_DEVICE}]")
if DEVICE_FALLBACK_NOTE:
    print(f"[!] Device note: {DEVICE_FALLBACK_NOTE}")

# Standardized Classes for Strawberry Ripeness & Anatomy
CLASSES = list(DEFAULT_CLASS_NAMES)
CLASS_INFO = [dict(c) for c in DEFAULT_CLASS_INFO]

# Model cache
MODELS = {}

def get_weight_path(filename):
    """Tìm file trọng số trong weights/ hoặc thư mục gốc."""
    p1 = os.path.join(WEIGHTS_DIR, filename)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(BASE_DIR, filename)
    if os.path.exists(p2):
        return p2
    return p1

def get_sam_model(model_key):
    if model_key not in MODELS:
        if model_key in ["sam3_vith", "sam2_large"]:
            weight_path = get_weight_path("sam2.1_l.pt")
        elif model_key == "sam2_base":
            weight_path = get_weight_path("sam2.1_b.pt")
        else: # sam2_tiny (default recommended)
            weight_path = get_weight_path("sam2.1_t.pt")

        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        print(f"[*] Loading SAM model [{model_key}] from {weight_path} onto {DEVICE}...")
        try:
            MODELS[model_key] = SAM(weight_path)
        except Exception:
            fallback_tiny = get_weight_path("sam2.1_t.pt")
            if weight_path != fallback_tiny and os.path.exists(fallback_tiny):
                print(f"[!] Fallback to {fallback_tiny}...")
                MODELS[model_key] = SAM(fallback_tiny)
            else:
                raise
    return MODELS[model_key]

# Lazy-loaded Zero-Shot Classifier (CLIP) for Auto Object Recognition
CLIP_MODEL = None
CLIP_PREPROCESS = None
CLIP_TEXT_FEATURES = None
CLIP_DEVICE = None

def clear_model_caches():
    """Giải phóng các model đang cache trên RAM/VRAM khi chuyển thiết bị hoặc benchmark."""
    global CLIP_MODEL, CLIP_PREPROCESS, CLIP_TEXT_FEATURES, CLIP_DEVICE
    MODELS.clear()
    CLIP_MODEL = None
    CLIP_PREPROCESS = None
    CLIP_TEXT_FEATURES = None
    CLIP_DEVICE = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

PROMPTS = [
    "an orange yellowish almost ripe strawberry fruit",
    "green strawberry calyx leaves sepals directly touching the fruit",
    "a long strawberry stem stalk or branch twig",
    "a pale green or white unripe strawberry fruit",
    "a bright red ripe juicy strawberry fruit"
]

def get_clip_classifier():
    global CLIP_MODEL, CLIP_PREPROCESS, CLIP_TEXT_FEATURES, CLIP_DEVICE
    if CLIP_MODEL is None or CLIP_DEVICE != DEVICE:
        import clip
        CLIP_MODEL = None
        CLIP_TEXT_FEATURES = None
        clip_dir = os.path.join(WEIGHTS_DIR, "clip")
        os.makedirs(clip_dir, exist_ok=True)
        weight_p = os.path.join(clip_dir, "ViT-B-32.pt")
        print(f"[*] Loading CLIP classifier from {weight_p} onto {DEVICE}...")
        if os.path.exists(weight_p):
            CLIP_MODEL, CLIP_PREPROCESS = clip.load(weight_p, device=DEVICE)
        else:
            CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE, download_root=clip_dir)
        tokens = clip.tokenize(PROMPTS).to(DEVICE)
        with torch.no_grad():
            text_feat = CLIP_MODEL.encode_text(tokens)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            CLIP_TEXT_FEATURES = text_feat
        CLIP_DEVICE = DEVICE
    return CLIP_MODEL, CLIP_PREPROCESS, CLIP_TEXT_FEATURES

def classify_crop(pil_crop, target_classes=None):
    """
    Tự động nhận diện class của vùng ảnh vẽ bao:
    Kết hợp giữa Zero-shot CLIP similarity và đặc trưng hình học (tỉ lệ dài/rộng) + màu sắc (HSV).
    """
    active_classes = list(target_classes) if target_classes else list(CLASSES)
    fallback_cls = active_classes[0] if active_classes else "calyx"
    try:
        import clip
        model, preprocess, default_text_feat = get_clip_classifier()

        img_in = preprocess(pil_crop).unsqueeze(0).to(DEVICE)
        use_default_strawberry = (
            len(active_classes) == len(CLASSES) and set(active_classes) == set(CLASSES)
        )

        if use_default_strawberry:
            text_feat = default_text_feat
            score_labels = CLASSES
        else:
            prompts = [f"a photo of {c}" for c in active_classes]
            tokens = clip.tokenize(prompts).to(DEVICE)
            with torch.no_grad():
                text_feat = model.encode_text(tokens)
                text_feat /= text_feat.norm(dim=-1, keepdim=True)
            score_labels = active_classes

        with torch.no_grad():
            img_feat = model.encode_image(img_in)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            sim = (100.0 * img_feat @ text_feat.T).softmax(dim=-1)[0].cpu().numpy()

        scores = dict(zip(score_labels, [float(s) for s in sim]))
        if use_default_strawberry:
            np_crop = np.array(pil_crop)
            if np_crop.size > 0:
                h, w = np_crop.shape[:2]
                aspect_ratio = max(w, h) / max(1, min(w, h))

                # Đặc trưng hình học: Cuống (stem) thường dài và mảnh
                if aspect_ratio >= 2.0:
                    scores['stem'] *= 2.5
                elif aspect_ratio < 1.3:
                    scores['stem'] *= 0.4

                # Phân tích dải màu HSV
                if len(np_crop.shape) == 3 and np_crop.shape[2] == 3:
                    hsv = cv2.cvtColor(np_crop, cv2.COLOR_RGB2HSV)
                    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
                    red_mask = ((H < 12) | (H > 160)) & (S > 45) & (V > 45)
                    red_pct = float(np.mean(red_mask))
                    if red_pct > 0.18:
                        scores['ripe'] *= 3.0
                        scores['unripe'] *= 0.1

        best_class = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = int((scores[best_class] / total) * 100) if total > 0 else 80
        return best_class, confidence
    except Exception as e:
        print(f"[!] Auto-classify exception: {e}")
        return fallback_cls, 50

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

def _run_sam_predict(model_id, img_path, **kwargs):
    """Chạy SAM với thiết bị đang chọn, tự động dự phòng về CPU nếu GPU lỗi hoặc thiếu VRAM."""
    model = get_sam_model(model_id)
    call_kwargs = dict(kwargs)
    call_kwargs["device"] = DEVICE
    try:
        return model(img_path, **call_kwargs)
    except Exception as e:
        if DEVICE == "cpu":
            raise
        print(f"[!] Inference exception on {DEVICE}: {e}. Retrying on CPU...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        call_kwargs["device"] = "cpu"
        return model(img_path, **call_kwargs)

@app.route("/api/system_info", methods=["GET"])
def system_info():
    with DEVICE_LOCK:
        info = get_hardware_info(
            DEVICE,
            DEVICE_NAME,
            PREFERRED_DEVICE,
            fallback_note=DEVICE_FALLBACK_NOTE,
        )
    return jsonify(info)

@app.route("/api/set_device", methods=["POST"])
def api_set_device():
    global DEVICE, DEVICE_NAME, PREFERRED_DEVICE, DEVICE_FALLBACK_NOTE
    data = request.json or {}
    target = str(data.get("device", "")).strip().lower()
    if target not in ("cpu", "cuda", "auto"):
        return jsonify({
            "success": False,
            "error": "Thiết bị không hợp lệ (chỉ chấp nhận 'cpu', 'cuda' hoặc 'auto').",
        }), 400

    with DEVICE_LOCK:
        if target == "cuda" and not torch.cuda.is_available():
            info = get_hardware_info(
                DEVICE,
                DEVICE_NAME,
                PREFERRED_DEVICE,
                fallback_note=DEVICE_FALLBACK_NOTE,
            )
            return jsonify({
                "success": False,
                "code": "CUDA_NOT_AVAILABLE",
                "error": (
                    f"Chưa thể bật GPU vì PyTorch hiện tại ({torch.__version__}) là bản CPU "
                    "(torch.cuda.is_available() == False). Hãy cài PyTorch bản CUDA trước rồi khởi động lại server."
                ),
                **info,
            }), 400

        new_dev, new_name, fallback_note = resolve_runtime_device(target)
        changed = (new_dev != DEVICE)
        DEVICE = new_dev
        DEVICE_NAME = new_name
        PREFERRED_DEVICE = target
        DEVICE_FALLBACK_NOTE = fallback_note
        save_device_preference(BASE_DIR, PREFERRED_DEVICE)

        if changed:
            clear_model_caches()
            print(f"[*] Switched inference device to: {DEVICE} ({DEVICE_NAME}) [pref={PREFERRED_DEVICE}]")

        info = get_hardware_info(
            DEVICE,
            DEVICE_NAME,
            PREFERRED_DEVICE,
            fallback_note=DEVICE_FALLBACK_NOTE,
        )
        return jsonify({"success": True, "switched": changed, **info})

@app.route("/api/benchmark_gpu", methods=["POST"])
def api_benchmark_gpu():
    data = request.json or {}
    include_sam_large = bool(data.get("include_sam_large", False))

    sample_image_path = None
    try:
        ctx = pm.get_active_context()
        img_dir = ctx.get("images_dir")
        if img_dir and os.path.isdir(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                if is_image_file(fname):
                    sample_image_path = os.path.join(img_dir, fname)
                    break
    except Exception:
        sample_image_path = None

    with DEVICE_LOCK:
        clear_model_caches()
        hw_info = get_hardware_info(
            DEVICE,
            DEVICE_NAME,
            PREFERRED_DEVICE,
            fallback_note=DEVICE_FALLBACK_NOTE,
        )
        report = run_device_benchmark(
            weights_dir=WEIGHTS_DIR,
            base_dir=BASE_DIR,
            hw_info=hw_info,
            sample_image_path=sample_image_path,
            include_sam_large=include_sam_large,
        )
        clear_model_caches()

    return jsonify(report)

@app.route("/api/classes", methods=["GET"])
def get_classes():
    ctx = pm.get_active_context()
    return jsonify({
        "classes": ctx["class_info"],
        "class_names": ctx["classes"],
        "project_id": ctx["id"],
        "project_name": ctx["name"],
    })

@app.route("/api/models", methods=["GET"])
def list_models():
    speed_gpu = "~0.05s (GPU) / ~1.5s (CPU)"
    speed_large = "~0.25s (GPU) / ~4-5s (CPU)"
    models = [
        {
            "id": "sam2_tiny",
            "name": "SAM 2.1 (Tiny) - Khuyên dùng (Siêu nhẹ & Nhanh)",
            "description": "Tối ưu cho cả CPU và GPU GTX 1650 4GB VRAM, phản hồi nhanh, không lo tràn VRAM",
            "size": "~75 MB",
            "speed": speed_gpu,
            "selected": True
        },
        {
            "id": "sam2_large",
            "name": "SAM 2.1 (Large) - Chi tiết cao",
            "description": "Mặt nạ đa giác độ phân giải cao, yêu cầu cấu hình và tài nguyên lớn hơn",
            "size": "~428 MB (224M Params)",
            "speed": speed_large,
            "selected": False
        },
        {
            "id": "sam3_vith",
            "name": "SAM 3 / ViT-H Masks",
            "description": "Bản mô phỏng ViT-Large segmentation",
            "size": "~428 MB+",
            "speed": speed_large,
            "selected": False
        }
    ]
    return jsonify(models)

@app.route("/api/images", methods=["GET"])
def list_images():
    ctx = pm.get_active_context()
    images_dir = ctx["images_dir"]
    ann_dir = ctx["annotations_dir"]
    splits_map = (ctx["manifest"] or {}).get("splits", {})
    files = [f for f in os.listdir(images_dir) if is_image_file(f)]
    result = []
    for f in sorted(files):
        ann_path = os.path.join(ann_dir, f"{f}.json")
        ann_count = 0
        if not os.path.exists(ann_path):
            with open(ann_path, 'w', encoding='utf-8') as af:
                json.dump({"annotations": []}, af, indent=2)
        else:
            try:
                with open(ann_path, 'r', encoding='utf-8') as af:
                    ann_data = json.load(af)
                    ann_count = len(ann_data.get('annotations', []))
            except Exception:
                ann_count = 0
        result.append({
            "filename": f,
            "url": f"/api/image/{quote(f)}?p={ctx['id']}",
            "annotations_count": ann_count,
            "split": splits_map.get(f)
        })
    return jsonify(result)

@app.route("/api/image/<path:filename>", methods=["GET"])
def get_image(filename):
    ctx = pm.get_active_context()
    safe_name = os.path.basename(filename)
    resp = send_from_directory(ctx["images_dir"], safe_name, max_age=0)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp

@app.route("/api/upload", methods=["POST"])
def upload_image():
    ctx = pm.get_active_context()
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    safe_name = os.path.basename(f.filename or "")
    if safe_name == "":
        return jsonify({"error": "Empty filename"}), 400
    save_path = os.path.join(ctx["images_dir"], safe_name)
    f.save(save_path)
    
    ann_path = os.path.join(ctx["annotations_dir"], f"{safe_name}.json")
    if not os.path.exists(ann_path):
        pm.sync_image_annotations(safe_name, [], update_last_image=False)

    return jsonify({"success": True, "filename": safe_name, "url": f"/api/image/{quote(safe_name)}?p={ctx['id']}"})

def _contour_to_clean_polygon(chosen, orig_w, orig_h, eps_ratio=0.0018, min_eps=0.8):
    """Làm mượt bậc thang pixel trên contour chính và rút gọn đỉnh bằng approxPolyDP."""
    single_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    cv2.drawContours(single_mask, [chosen], -1, 255, thickness=cv2.FILLED)
    blurred = cv2.GaussianBlur(single_mask, (3, 3), 0)
    _, smooth_bin = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    smooth_cnts, _ = cv2.findContours(smooth_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if smooth_cnts:
        chosen = max(smooth_cnts, key=cv2.contourArea)

    peri = cv2.arcLength(chosen, True)
    eps = max(min_eps, eps_ratio * peri)
    approx = cv2.approxPolyDP(chosen, eps, True)
    pts = approx.reshape(-1, 2).tolist()
    if len(pts) < 3:
        pts = chosen.reshape(-1, 2).tolist()
    if len(pts) < 3:
        return []
    return [[round(float(x), 1), round(float(y), 1)] for x, y in pts]


def _extract_clean_polygons_from_mask(mask_np, orig_w, orig_h, bbox=None, lasso_polygon=None, point=None):
    """Trích xuất đường bao polygon sạch từ mask nhị phân của SAM, không nối bắc cầu các đảo nhiễu."""
    mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255 if mask_np.dtype != np.uint8 else ((mask_np > 0).astype(np.uint8) * 255)
    if mask_u8.shape[0] != orig_h or mask_u8.shape[1] != orig_w:
        mask_u8 = cv2.resize(mask_u8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if np.count_nonzero(mask_u8) < 4:
        return []

    raw_mask_u8 = mask_u8
    if lasso_polygon is not None and len(lasso_polygon) >= 3:
        lasso_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        lasso_pts = np.round(np.array(lasso_polygon, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(lasso_mask, [lasso_pts], 255)
        k_lasso = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lasso_mask = cv2.dilate(lasso_mask, k_lasso, iterations=1)
        clipped = cv2.bitwise_and(mask_u8, lasso_mask)
        if np.count_nonzero(clipped) >= 12:
            mask_u8 = clipped
    elif bbox is not None and len(bbox) == 4:
        bx1, by1, bx2, by2 = [float(v) for v in bbox]
        pad = 3
        x1 = max(0, min(orig_w, int(round(min(bx1, bx2))) - pad))
        y1 = max(0, min(orig_h, int(round(min(by1, by2))) - pad))
        x2 = max(0, min(orig_w, int(round(max(bx1, bx2))) + pad))
        y2 = max(0, min(orig_h, int(round(max(by1, by2))) + pad))
        if x2 > x1 and y2 > y1:
            bbox_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            bbox_mask[y1:y2, x1:x2] = 255
            clipped = cv2.bitwise_and(mask_u8, bbox_mask)
            if np.count_nonzero(clipped) >= 12:
                mask_u8 = clipped

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k_open, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=1)
    if np.count_nonzero(cleaned) < 12:
        cleaned = mask_u8

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contours, _ = cv2.findContours(raw_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    valid_cnts = [c for c in contours if cv2.contourArea(c) >= 16.0]
    if not valid_cnts:
        valid_cnts = list(contours)

    if point is not None and len(point) == 2:
        px, py = float(point[0]), float(point[1])
        containing = [c for c in valid_cnts if cv2.pointPolygonTest(c, (px, py), False) >= 0]
        if containing:
            chosen = max(containing, key=cv2.contourArea)
        else:
            chosen = min(valid_cnts, key=lambda c: abs(cv2.pointPolygonTest(c, (px, py), True)))
    else:
        chosen = max(valid_cnts, key=cv2.contourArea)

    poly = _contour_to_clean_polygon(chosen, orig_w, orig_h)
    return [poly] if len(poly) >= 3 else []


def _sanitize_corrupted_polygon(poly, img_w, img_h):
    """Phát hiện và làm sạch đa giác bị lỗi bắc cầu (merge_multi_segment) khi đã lưu trước đó."""
    if not poly or len(poly) < 6 or img_w <= 0 or img_h <= 0:
        return poly, False

    pts_tuples = [(round(float(p[0]), 2), round(float(p[1]), 2)) for p in poly]
    has_dup_pts = len(set(pts_tuples)) < len(pts_tuples)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    pts_i32 = np.round(np.array(poly, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts_i32], 255)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    multi_island = len(cnts) > 1

    if not has_dup_pts and not multi_island:
        return poly, False

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close, iterations=1)
    cnts_clean, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_clean:
        cnts_clean = cnts
    if not cnts_clean:
        return poly, False

    chosen = max(cnts_clean, key=cv2.contourArea)
    clean_poly = _contour_to_clean_polygon(chosen, img_w, img_h)
    if len(clean_poly) < 3:
        return poly, False
    return clean_poly, True


@app.route("/api/segment", methods=["POST"])
def segment():
    ctx = pm.get_active_context()
    data = request.json or {}
    image_name = data.get("image_name")
    model_id = data.get("model_id", "sam2_tiny")
    bbox = data.get("bbox") # [x1, y1, x2, y2]
    lasso_polygon = data.get("lasso_polygon") # [[x, y], ...]
    point = data.get("point") # [x, y]
    default_cls = ctx["classes"][0] if ctx["classes"] else "calyx"
    label_class = data.get("class_name", default_cls)
    auto_classify = data.get("auto_classify", True)

    if not image_name:
        return jsonify({"error": "Missing image_name"}), 400

    img_path = os.path.join(ctx["images_dir"], os.path.basename(image_name))
    if not os.path.exists(img_path):
        return jsonify({"error": "Image not found"}), 404

    with Image.open(img_path) as im:
        orig_w, orig_h = im.size
        if not bbox and lasso_polygon and len(lasso_polygon) >= 3:
            lxs = [float(p[0]) for p in lasso_polygon]
            lys = [float(p[1]) for p in lasso_polygon]
            bbox = [min(lxs), min(lys), max(lxs), max(lys)]

        confidence = None
        auto_classified = False

        # Tự động nhận dạng vật thể trong vùng bao (Auto-Classification)
        if auto_classify and bbox:
            try:
                x1, y1, x2, y2 = bbox
                crop_box = (
                    int(max(0, min(x1, x2))),
                    int(max(0, min(y1, y2))),
                    int(min(orig_w, max(x1, x2))),
                    int(min(orig_h, max(y1, y2)))
                )
                if (crop_box[2] - crop_box[0] > 8) and (crop_box[3] - crop_box[1] > 8):
                    crop = im.crop(crop_box)
                    detected_class, conf = classify_crop(crop, target_classes=ctx["classes"])
                    label_class = detected_class
                    confidence = conf
                    auto_classified = True
                    print(f"[*] Tự động nhận diện vật thể: {label_class} ({confidence}%)")
            except Exception as e:
                print(f"[!] Auto-classify failed: {e}")

    sam_kwargs = {}
    if bbox:
        sam_kwargs["bboxes"] = [bbox]
    elif point:
        sam_kwargs["points"] = [[point[0], point[1]]]
        sam_kwargs["labels"] = [1]
    else:
        return jsonify({"error": "Provide bbox or point"}), 400

    try:
        results = _run_sam_predict(model_id, img_path, **sam_kwargs)
    except Exception as e:
        print("[!] SAM segmentation error:", e)
        return jsonify({"error": str(e)}), 500

    try:
        polygons = []
        for r in results:
            if r.masks is not None and r.masks.data is not None and len(r.masks.data) > 0:
                for mask_tensor in r.masks.data:
                    mask_np = mask_tensor.detach().cpu().numpy()
                    extracted = _extract_clean_polygons_from_mask(
                        mask_np,
                        orig_w,
                        orig_h,
                        bbox=bbox,
                        lasso_polygon=lasso_polygon,
                        point=point,
                    )
                    polygons.extend(extracted)

        if not polygons and bbox:
            x1, y1, x2, y2 = bbox
            polygons = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]

        return jsonify({
            "success": True,
            "polygons": polygons,
            "class_name": label_class,
            "confidence": confidence,
            "auto_classified": auto_classified
        })
    except Exception as e:
        print("[!] Segmentation parsing error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/edit_polygon", methods=["POST"])
def edit_polygon():
    """
    Chỉnh sửa polygon:
    - 'erase': Xóa bớt phần thừa bằng các nét tẩy (vòng tròn / đường nối các điểm kéo tẩy)
    - 'add': Đắp thêm phần thiếu bằng cách hợp nhất (union) polygon vẽ bù
    """
    try:
        ctx = pm.get_active_context()
        data = request.json or {}
        polygon = data.get("polygon", [])
        operation = data.get("operation", "erase")  # 'erase' or 'add'
        img_name = data.get("image_name", "")
        img_w = int(data.get("image_width", 0))
        img_h = int(data.get("image_height", 0))

        if not polygon or len(polygon) < 3:
            return jsonify({"error": "Polygon không hợp lệ hoặc ít hơn 3 điểm"}), 400

        # Xác định kích thước ảnh thực tế
        if (img_w <= 0 or img_h <= 0) and img_name:
            ipath = os.path.join(ctx["images_dir"], os.path.basename(img_name))
            if os.path.exists(ipath):
                with Image.open(ipath) as im:
                    img_w, img_h = im.size

        if img_w <= 0 or img_h <= 0:
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            img_w = max(int(max(xs)) + 50, 100)
            img_h = max(int(max(ys)) + 50, 100)

        # Tạo mask nhị phân từ polygon ban đầu
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        poly_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [poly_np], 255)

        if operation == "erase":
            strokes = data.get("strokes", [])
            for s in strokes:
                r = max(1, int(s.get("radius", 15)))
                if "x1" in s and "x2" in s:
                    x1, y1 = int(round(s["x1"])), int(round(s["y1"]))
                    x2, y2 = int(round(s["x2"])), int(round(s["y2"]))
                    cv2.line(mask, (x1, y1), (x2, y2), 0, thickness=r * 2)
                    cv2.circle(mask, (x1, y1), r, 0, -1)
                    cv2.circle(mask, (x2, y2), r, 0, -1)
                elif "x" in s and "y" in s:
                    x, y = int(round(s["x"])), int(round(s["y"]))
                    cv2.circle(mask, (x, y), r, 0, -1)

        elif operation == "add":
            add_poly = data.get("add_polygon", [])
            if add_poly and len(add_poly) >= 3:
                add_np = np.array(add_poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [add_np], 255)

        # Trích xuất contours từ mask đã chỉnh sửa
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return jsonify({"success": True, "polygons": [], "deleted": True})

        new_polys = []
        sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        max_area = cv2.contourArea(sorted_cnts[0]) if sorted_cnts else 0.0
        min_keep_area = max(15.0, 0.015 * max_area)
        for c in sorted_cnts:
            area = cv2.contourArea(c)
            if area < min_keep_area:
                continue
            eps = max(0.5, 0.0015 * cv2.arcLength(c, True))
            approx = cv2.approxPolyDP(c, eps, True)
            pts = approx.reshape(-1, 2).tolist()
            if len(pts) >= 3:
                new_polys.append([[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pts])

        if not new_polys and len(sorted_cnts) > 0:
            pts = sorted_cnts[0].reshape(-1, 2).tolist()
            if len(pts) >= 3:
                new_polys.append([[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pts])

        return jsonify({
            "success": True,
            "polygons": new_polys,
            "deleted": len(new_polys) == 0
        })
    except Exception as e:
        print("[!] Error in edit_polygon:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/annotations/<path:filename>", methods=["GET", "POST"])
def handle_annotations(filename):
    ctx = pm.get_active_context()
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        return jsonify({"error": "Invalid filename"}), 400
    ann_path = os.path.join(ctx["annotations_dir"], f"{safe_name}.json")
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        try:
            res = pm.sync_image_annotations(safe_name, data.get("annotations", []))
            return jsonify(res)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    else:
        pm.record_last_image(safe_name)
        if os.path.exists(ann_path):
            with open(ann_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        return jsonify({"annotations": []})

@app.route("/api/export", methods=["GET", "POST"])
def export_yolo():
    try:
        res = pm.export_active_yolo()
        return jsonify(res)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print("[!] Export error:", e)
        return jsonify({"error": str(e)}), 500


# ==============================================================================
# Dataset Project Management APIs
# ==============================================================================

@app.route("/api/projects", methods=["GET"])
def api_list_projects():
    try:
        return jsonify(pm.list_projects_state())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/browse_folder", methods=["POST"])
def api_browse_folder():
    data = request.json or {}
    title = str(data.get("title", "Chọn thư mục")).replace('"', "'")
    initial_dir = data.get("initial_dir") or BASE_DIR
    if not os.path.isdir(initial_dir):
        initial_dir = BASE_DIR

    script = (
        "import sys, tkinter as tk; from tkinter import filedialog; "
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True); r.update(); "
        f"p = filedialog.askdirectory(title={json.dumps(title)}, initialdir={json.dumps(initial_dir)}); "
        "r.destroy(); "
        "sys.stdout.buffer.write((p or '').encode('utf-8'))"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=120)
        selected = proc.stdout.decode("utf-8", errors="ignore").strip()
        if not selected:
            return jsonify({"success": True, "cancelled": True, "path": ""})
        return jsonify({"success": True, "cancelled": False, "path": os.path.abspath(selected)})
    except Exception as e:
        return jsonify({"success": False, "error": f"Không mở được hộp thoại chọn thư mục: {e}"}), 500


@app.route("/api/projects/analyze_path", methods=["POST"])
def api_analyze_local_path():
    data = request.json or {}
    source_dir = (data.get("source_dir") or "").strip()
    try:
        scan = inspect_dataset_folder(source_dir)
        scan.pop("entries", None)
        scan["source_dir"] = os.path.abspath(source_dir)
        scan["suggested_name"] = os.path.basename(os.path.abspath(source_dir)) or "new_dataset_project"
        return jsonify({"success": True, **scan})
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/projects/create_from_path", methods=["POST"])
def api_create_project_from_path():
    data = request.json or {}
    source_dir = (data.get("source_dir") or "").strip()
    save_dir = (data.get("save_dir") or "").strip()
    project_name = (data.get("project_name") or "").strip()
    custom_classes = data.get("classes")
    use_default_classes = bool(data.get("use_default_classes", False))
    try:
        created = pm.create_project_from_local_folder(
            source_dir=source_dir,
            save_dir=save_dir,
            project_name=project_name,
            custom_classes=custom_classes,
            use_default_classes=use_default_classes,
        )
        state = pm.list_projects_state()
        return jsonify({"success": True, **created, **state})
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        print("[!] create_project_from_path error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/projects/upload_init", methods=["POST"])
def api_project_upload_init():
    data = request.json or {}
    save_dir = (data.get("save_dir") or "").strip()
    project_name = (data.get("project_name") or "").strip()
    source_name = (data.get("source_name") or "").strip()
    try:
        res = pm.init_upload_session(save_dir=save_dir, project_name=project_name, source_name=source_name)
        return jsonify({"success": True, **res})
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/projects/upload_chunk", methods=["POST"])
def api_project_upload_chunk():
    session_id = request.form.get("session_id", "")
    files = request.files.getlist("files")
    rel_paths = request.form.getlist("relative_paths")
    paired = []
    for idx, fobj in enumerate(files):
        rpath = rel_paths[idx] if idx < len(rel_paths) else fobj.filename
        paired.append((rpath, fobj))
    try:
        res = pm.add_upload_chunk(session_id, paired)
        return jsonify({"success": True, **res})
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/projects/upload_finalize", methods=["POST"])
def api_project_upload_finalize():
    data = request.json or {}
    session_id = data.get("session_id", "")
    custom_classes = data.get("classes")
    use_default_classes = bool(data.get("use_default_classes", False))
    try:
        created = pm.finalize_upload_session(
            session_id=session_id,
            custom_classes=custom_classes,
            use_default_classes=use_default_classes,
        )
        state = pm.list_projects_state()
        return jsonify({"success": True, **created, **state})
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        print("[!] upload_finalize error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/projects/upload_cancel", methods=["POST"])
def api_project_upload_cancel():
    data = request.json or {}
    session_id = data.get("session_id", "")
    pm.cancel_upload_session(session_id)
    return jsonify({"success": True})


@app.route("/api/projects/switch", methods=["POST"])
def api_switch_project():
    data = request.json or {}
    project_id = data.get("project_id")
    project_dir = data.get("project_dir")
    try:
        state = pm.switch_project(project_id=project_id, project_dir=project_dir)
        return jsonify({"success": True, **state})
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/projects/open_folder", methods=["POST"])
def api_open_project_folder():
    data = request.json or {}
    project_dir = (data.get("project_dir") or "").strip()
    try:
        state = pm.open_existing_project_folder(project_dir)
        return jsonify({"success": True, **state})
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/projects/remove", methods=["POST"])
def api_remove_project():
    data = request.json or {}
    project_id = data.get("project_id", "")
    try:
        state = pm.remove_from_registry(project_id)
        return jsonify({"success": True, **state})
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print(f"[*] Starting RoboLabel AI on http://127.0.0.1:5050 ...")
    app.run(host="127.0.0.1", port=5050, debug=False)
