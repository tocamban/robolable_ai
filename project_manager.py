import os
import re
import json
import uuid
import shutil
import random
from datetime import datetime
from collections import Counter
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ALLOWED_IMPORT_EXTS = IMAGE_EXTS + (".txt", ".yaml", ".yml")
IGNORED_TXT_NAMES = {
    "classes.txt",
    "obj.names",
    "notes.txt",
    "readme.txt",
    "requirements.txt",
    "train.txt",
    "val.txt",
    "valid.txt",
    "test.txt",
    "_darknet.labels",
}
SKIP_DIRS = {".git", "__pycache__", "node_modules", "_import_tmp", ".venv", "venv"}

DEFAULT_CLASS_INFO = [
    {"name": "almost_ripe", "color": "#f97316", "label": "Gần chín (Almost Ripe)"},
    {"name": "calyx", "color": "#a855f7", "label": "Đài hoa (Calyx)"},
    {"name": "stem", "color": "#eab308", "label": "Cuống quả (Stem)"},
    {"name": "unripe", "color": "#22c55e", "label": "Chưa chín (Unripe)"},
    {"name": "ripe", "color": "#ef4444", "label": "Chín mọng (Ripe)"},
]
DEFAULT_CLASS_NAMES = [c["name"] for c in DEFAULT_CLASS_INFO]
DEFAULT_CLASS_MAP = {c["name"]: c for c in DEFAULT_CLASS_INFO}

PALETTE_COLORS = [
    "#7c3aed", "#ef4444", "#22c55e", "#f97316", "#eab308",
    "#3b82f6", "#a855f7", "#ec4899", "#14b8a6", "#0ea5e9",
    "#84cc16", "#f59e0b", "#6366f1", "#d946ef", "#fb7185",
]


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTS)


def sanitize_folder_name(name):
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (name or "").strip())
    cleaned = cleaned.strip(". ")
    return cleaned or f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def atomic_write_text(path, content):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def atomic_write_json(path, data):
    content = json.dumps(data, ensure_ascii=False, indent=2)
    atomic_write_text(path, content)


def _strip_yaml_comment(line):
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _unquote(val):
    v = val.strip()
    if len(v) >= 2 and ((v[0] == "'" and v[-1] == "'") or (v[0] == '"' and v[-1] == '"')):
        return v[1:-1]
    return v


def parse_yolo_yaml(yaml_text):
    """Parse common YOLO data.yaml structures without requiring external PyYAML."""
    result = {"names": [], "nc": None, "has_yolo_keys": False}
    if not yaml_text:
        return result

    lines = yaml_text.lstrip("\ufeff").splitlines()
    i = 0
    while i < len(lines):
        raw_line = _strip_yaml_comment(lines[i]).rstrip()
        stripped = raw_line.strip()
        if not stripped:
            i += 1
            continue

        if re.match(r"^(train|val|test|path)\s*:", stripped):
            result["has_yolo_keys"] = True

        nc_match = re.match(r"^nc\s*:\s*(\d+)\s*$", stripped)
        if nc_match:
            result["nc"] = int(nc_match.group(1))
            result["has_yolo_keys"] = True
            i += 1
            continue

        names_match = re.match(r"^names\s*:\s*(.*)$", raw_line.lstrip())
        if names_match and raw_line.startswith("names"):
            result["has_yolo_keys"] = True
            rest = names_match.group(1).strip()

            # Case 1: Inline list e.g. names: ['a', 'b', 'c']
            if rest.startswith("["):
                buf = rest
                while "]" not in buf and i + 1 < len(lines):
                    i += 1
                    buf += " " + _strip_yaml_comment(lines[i]).strip()
                inner = buf[buf.find("[") + 1:]
                if "]" in inner:
                    inner = inner[:inner.find("]")]
                items = [_unquote(part) for part in inner.split(",") if part.strip()]
                result["names"] = [it for it in items if it != ""]
                i += 1
                continue

            # Case 2: Inline dict e.g. names: {0: 'a', 1: 'b'}
            if rest.startswith("{"):
                buf = rest
                while "}" not in buf and i + 1 < len(lines):
                    i += 1
                    buf += " " + _strip_yaml_comment(lines[i]).strip()
                inner = buf[buf.find("{") + 1:]
                if "}" in inner:
                    inner = inner[:inner.find("}")]
                parsed_map = {}
                for part in inner.split(","):
                    if ":" in part:
                        k_str, v_str = part.split(":", 1)
                        try:
                            parsed_map[int(_unquote(k_str))] = _unquote(v_str)
                        except ValueError:
                            pass
                if parsed_map:
                    max_idx = max(parsed_map.keys())
                    result["names"] = [parsed_map.get(idx, f"class_{idx}") for idx in range(max_idx + 1)]
                i += 1
                continue

            # Case 3: Indented list or numeric mapping on following lines
            base_indent = len(raw_line) - len(raw_line.lstrip())
            collected_list = []
            collected_map = {}
            j = i + 1
            while j < len(lines):
                sub_raw = _strip_yaml_comment(lines[j]).rstrip()
                sub_strip = sub_raw.strip()
                if not sub_strip:
                    j += 1
                    continue
                sub_indent = len(sub_raw) - len(sub_raw.lstrip())
                if sub_indent <= base_indent and not sub_strip.startswith("-"):
                    break
                dash_m = re.match(r"^-\s+(.+)$", sub_strip)
                map_m = re.match(r"^(\d+)\s*:\s*(.+)$", sub_strip)
                if dash_m:
                    collected_list.append(_unquote(dash_m.group(1)))
                elif map_m:
                    collected_map[int(map_m.group(1))] = _unquote(map_m.group(2))
                j += 1

            if collected_map:
                max_idx = max(collected_map.keys())
                result["names"] = [collected_map.get(idx, f"class_{idx}") for idx in range(max_idx + 1)]
            elif collected_list:
                result["names"] = collected_list
            i = j
            continue

        i += 1

    return result


def parse_classes_txt(text):
    names = []
    for line in (text or "").lstrip("\ufeff").splitlines():
        val = line.strip()
        if val:
            names.append(val)
    return names


def build_class_info(class_names):
    clean_names = []
    seen = set()
    for raw in (class_names or []):
        name = str(raw).strip()
        if name and name not in seen:
            clean_names.append(name)
            seen.add(name)

    if not clean_names:
        clean_names = list(DEFAULT_CLASS_NAMES)

    info = []
    for idx, name in enumerate(clean_names):
        if name in DEFAULT_CLASS_MAP:
            info.append(dict(DEFAULT_CLASS_MAP[name]))
        else:
            info.append({
                "name": name,
                "color": PALETTE_COLORS[idx % len(PALETTE_COLORS)],
                "label": name,
            })
    return clean_names, info


def _detect_split_from_rel(rel_path):
    parts = [p.lower() for p in rel_path.replace("\\", "/").split("/")[:-1]]
    for p in parts:
        if p == "train":
            return "train"
        if p in ("val", "valid", "validation"):
            return "val"
        if p == "test":
            return "test"
    return None


def _find_matching_label(img_rel, txt_by_rel, txt_by_rel_lower, txt_by_stem_lower):
    norm_rel = img_rel.replace("\\", "/")
    parts = norm_rel.split("/")
    stem = os.path.splitext(parts[-1])[0]

    candidates = []
    # 1. Sibling .txt in same folder
    sibling = "/".join(parts[:-1] + [f"{stem}.txt"]) if len(parts) > 1 else f"{stem}.txt"
    candidates.append(sibling)

    # 2 & 3. Swap last directory named 'images' -> 'labels' or 'image' -> 'label'
    for src_dir, dst_dir in (("images", "labels"), ("image", "label")):
        for idx in range(len(parts) - 2, -1, -1):
            if parts[idx].lower() == src_dir:
                swapped = list(parts[:-1])
                swapped[idx] = dst_dir
                candidates.append("/".join(swapped + [f"{stem}.txt"]))
                break

    # 4. Check '<parent>/labels/<stem>.txt' and top-level 'labels/' layouts
    if len(parts) >= 2:
        candidates.append("/".join(parts[:-1] + ["labels", f"{stem}.txt"]))
        candidates.append("labels/" + "/".join(parts[:-1] + [f"{stem}.txt"]))
    candidates.append(f"labels/{stem}.txt")

    for cand in candidates:
        if cand in txt_by_rel:
            return cand
        low = cand.lower()
        if low in txt_by_rel_lower:
            return txt_by_rel_lower[low]

    # 5. Unique stem match fallback when unambiguous (e.g., labels stored in a separate folder)
    stem_matches = txt_by_stem_lower.get(stem.lower(), [])
    if len(stem_matches) == 1:
        return stem_matches[0]
    elif len(stem_matches) > 1:
        img_split = _detect_split_from_rel(norm_rel)
        if img_split:
            split_matches = [m for m in stem_matches if _detect_split_from_rel(m) == img_split]
            if len(split_matches) == 1:
                return split_matches[0]
    return None


def inspect_dataset_folder(folder_path):
    """
    Scan a folder safely (read-only) to distinguish:
    - Plain image folder ('images_only')
    - YOLO Detection dataset ('yolo_detection')
    - YOLO Segmentation dataset ('yolo_segmentation')
    - Mixed YOLO Detection + Segmentation ('yolo_mixed')
    - Empty YOLO structure ('yolo_empty')
    """
    if not folder_path or not os.path.isdir(folder_path):
        raise ValueError("Thư mục nguồn không tồn tại.")

    image_rels = []
    txt_rels = []
    yaml_rels = []
    classes_txt_rels = []

    for root, dirs, files in os.walk(folder_path):
        dirs[:] = sorted([d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")])
        rel_dir = os.path.relpath(root, folder_path).replace("\\", "/")
        for fname in sorted(files):
            if fname.startswith("."):
                continue
            rel = fname if rel_dir == "." else f"{rel_dir}/{fname}"
            low = fname.lower()
            if is_image_file(fname):
                image_rels.append(rel)
            elif low.endswith(".txt"):
                if low in ("classes.txt", "obj.names"):
                    classes_txt_rels.append(rel)
                elif low not in IGNORED_TXT_NAMES:
                    txt_rels.append(rel)
            elif low.endswith((".yaml", ".yml")):
                yaml_rels.append(rel)

    # Sort YAML candidates by preference (data.yaml / dataset.yaml, then shallowest path)
    yaml_rels.sort(key=lambda r: (
        0 if os.path.basename(r).lower() in ("data.yaml", "dataset.yaml") else 1,
        r.count("/"),
        r
    ))

    yaml_info = {"names": [], "nc": None, "has_yolo_keys": False}
    yaml_source_rel = None
    for y_rel in yaml_rels:
        try:
            with open(os.path.join(folder_path, y_rel), "r", encoding="utf-8", errors="ignore") as yf:
                parsed = parse_yolo_yaml(yf.read())
            if parsed["names"] or parsed["has_yolo_keys"]:
                yaml_info = parsed
                yaml_source_rel = y_rel
                break
        except OSError:
            pass

    classes_txt_names = []
    classes_txt_rel = None
    if not yaml_info["names"] and classes_txt_rels:
        classes_txt_rels.sort(key=lambda r: (r.count("/"), r))
        try:
            with open(os.path.join(folder_path, classes_txt_rels[0]), "r", encoding="utf-8", errors="ignore") as cf:
                classes_txt_names = parse_classes_txt(cf.read())
                if classes_txt_names:
                    classes_txt_rel = classes_txt_rels[0]
        except OSError:
            pass

    txt_by_rel = {r: r for r in txt_rels}
    txt_by_rel_lower = {r.lower(): r for r in txt_rels}
    txt_by_stem_lower = {}
    for r in txt_rels:
        st = os.path.splitext(os.path.basename(r))[0].lower()
        txt_by_stem_lower.setdefault(st, []).append(r)

    # Collision-free stored names in project images/
    basenames_lower = Counter(os.path.basename(r).lower() for r in image_rels)
    used_stored_lower = set()

    entries = []
    matched_labels_count = 0
    non_empty_label_files = 0
    detection_count = 0
    segmentation_count = 0
    invalid_lines = 0
    max_class_id = -1
    splits_summary = Counter()

    for img_rel in image_rels:
        base = os.path.basename(img_rel)
        split = _detect_split_from_rel(img_rel)
        splits_summary[split or "unsplit"] += 1

        if basenames_lower[base.lower()] == 1 and base.lower() not in used_stored_lower:
            stored_name = base
        else:
            parent_parts = [p for p in img_rel.split("/")[:-1] if p not in (".", "")]
            prefix = split or (parent_parts[-1] if parent_parts else "img")
            prefix = re.sub(r'[<>:"/\\|?*\s]+', "_", prefix)
            candidate = f"{prefix}__{base}"
            stem_c, ext_c = os.path.splitext(candidate)
            n = 2
            while candidate.lower() in used_stored_lower:
                candidate = f"{stem_c}_{n}{ext_c}"
                n += 1
            stored_name = candidate

        used_stored_lower.add(stored_name.lower())
        label_rel = _find_matching_label(img_rel, txt_by_rel, txt_by_rel_lower, txt_by_stem_lower)

        if label_rel:
            matched_labels_count += 1
            try:
                with open(os.path.join(folder_path, label_rel), "r", encoding="utf-8", errors="ignore") as lf:
                    lines = [ln.strip() for ln in lf.readlines() if ln.strip()]
                if lines:
                    non_empty_label_files += 1
                for ln in lines:
                    parts = ln.split()
                    if len(parts) < 5:
                        invalid_lines += 1
                        continue
                    try:
                        cid = int(float(parts[0]))
                        coords = [float(v) for v in parts[1:]]
                    except ValueError:
                        invalid_lines += 1
                        continue
                    if cid < 0:
                        invalid_lines += 1
                        continue
                    max_class_id = max(max_class_id, cid)
                    if len(coords) == 4:
                        detection_count += 1
                    elif len(coords) >= 6 and len(coords) % 2 == 0:
                        segmentation_count += 1
                    else:
                        invalid_lines += 1
            except OSError:
                invalid_lines += 1

        entries.append({
            "img_rel": img_rel,
            "label_rel": label_rel,
            "split": split,
            "stored_name": stored_name,
        })

    # Determine dataset type
    if segmentation_count > 0 and detection_count == 0:
        dataset_format = "yolo_segmentation"
        format_label = "Dataset YOLO Segmentation"
    elif detection_count > 0 and segmentation_count == 0:
        dataset_format = "yolo_detection"
        format_label = "Dataset YOLO Detection"
    elif detection_count > 0 and segmentation_count > 0:
        dataset_format = "yolo_mixed"
        format_label = "Dataset YOLO (Box + Polygon)"
    elif matched_labels_count > 0 or yaml_source_rel is not None or classes_txt_rel is not None:
        dataset_format = "yolo_empty"
        format_label = "Dataset YOLO (Chưa có nhãn đối tượng)"
    else:
        dataset_format = "images_only"
        format_label = "Thư mục ảnh thông thường"

    # Determine resolved class names
    if yaml_info["names"]:
        detected_classes = list(yaml_info["names"])
        classes_source = yaml_source_rel or "data.yaml"
    elif classes_txt_names:
        detected_classes = list(classes_txt_names)
        classes_source = classes_txt_rel or "classes.txt"
    elif max_class_id >= 0:
        needed = max_class_id + 1
        if yaml_info["nc"]:
            needed = max(needed, yaml_info["nc"])
        detected_classes = [f"class_{idx}" for idx in range(needed)]
        classes_source = "label_ids"
    else:
        detected_classes = list(DEFAULT_CLASS_NAMES)
        classes_source = "default"

    if max_class_id >= len(detected_classes):
        for idx in range(len(detected_classes), max_class_id + 1):
            detected_classes.append(f"class_{idx}")

    return {
        "total_images": len(image_rels),
        "matched_label_files": matched_labels_count,
        "non_empty_label_files": non_empty_label_files,
        "detection_objects": detection_count,
        "segmentation_objects": segmentation_count,
        "total_objects": detection_count + segmentation_count,
        "invalid_lines": invalid_lines,
        "dataset_format": dataset_format,
        "format_label": format_label,
        "detected_classes": detected_classes,
        "classes_source": classes_source,
        "splits": dict(splits_summary),
        "yaml_rel": yaml_source_rel,
        "classes_txt_rel": classes_txt_rel,
        "entries": entries,
    }


def parse_yolo_label_file_to_annotations(label_path, img_w, img_h, class_names):
    """Convert normalized YOLO Detection or Segmentation lines to pixel-coordinate RoboLabel annotations."""
    annotations = []
    if not label_path or not os.path.exists(label_path) or img_w <= 0 or img_h <= 0:
        return annotations

    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue

        if cid < 0:
            continue
        if cid >= len(class_names):
            for idx in range(len(class_names), cid + 1):
                class_names.append(f"class_{idx}")
        cname = class_names[cid]

        # Detect normalized vs pixel coords defensively
        all_norm = all(-0.05 <= v <= 1.05 for v in vals)

        if len(vals) == 4:
            xc, yc, bw, bh = vals
            if all_norm:
                xc *= img_w
                yc *= img_h
                bw *= img_w
                bh *= img_h
            x1 = round(max(0.0, min(float(img_w), xc - bw / 2.0)), 1)
            y1 = round(max(0.0, min(float(img_h), yc - bh / 2.0)), 1)
            x2 = round(max(0.0, min(float(img_w), xc + bw / 2.0)), 1)
            y2 = round(max(0.0, min(float(img_h), yc + bh / 2.0)), 1)
            if x2 <= x1 or y2 <= y1:
                continue
            annotations.append({
                "class_name": cname,
                "type": "bbox",
                "bbox": [x1, y1, x2, y2],
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            })
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            pts = []
            for k in range(0, len(vals), 2):
                px = vals[k]
                py = vals[k + 1]
                if all_norm:
                    px = max(0.0, min(1.0, px)) * img_w
                    py = max(0.0, min(1.0, py)) * img_h
                else:
                    px = max(0.0, min(float(img_w), px))
                    py = max(0.0, min(float(img_h), py))
                pts.append([round(px, 1), round(py, 1)])
            if len(pts) >= 3:
                annotations.append({
                    "class_name": cname,
                    "type": "polygon",
                    "polygon": pts,
                })

    return annotations


def normalize_group_id(val):
    """Normalize an annotation group_id to a non-negative int, non-empty str, or None."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val if val >= 0 else None
    if isinstance(val, float):
        return int(val) if val.is_integer() and val >= 0 else None
    s = str(val).strip()
    if not s:
        return None
    try:
        n = int(s)
        return n if n >= 0 else None
    except ValueError:
        return s


def sanitize_annotations_list(annotations_list):
    """Return a clean copy of annotation dicts with normalized class_name and optional group_id."""
    cleaned = []
    for item in (annotations_list or []):
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        if not entry.get("class_name") and entry.get("label"):
            entry["class_name"] = str(entry["label"]).strip()
        gid = normalize_group_id(entry.get("group_id"))
        if gid is None:
            entry.pop("group_id", None)
        else:
            entry["group_id"] = gid
        cleaned.append(entry)
    return cleaned


def _is_axis_aligned_rect(poly):
    if not poly or len(poly) != 4:
        return False
    xs = sorted(set(round(float(p[0]), 2) for p in poly))
    ys = sorted(set(round(float(p[1]), 2) for p in poly))
    return len(xs) == 2 and len(ys) == 2


def _extract_clamped_bbox(item, img_w, img_h):
    """Return clamped (x1, y1, x2, y2) for bbox annotations, or None if not a valid box."""
    ann_type = item.get("type")
    bbox = item.get("bbox")
    poly = item.get("polygon", [])
    is_box = (ann_type == "bbox") or (bbox is not None and _is_axis_aligned_rect(poly))
    if not is_box:
        return None
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    elif poly and len(poly) >= 4:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    else:
        return None
    x1, x2 = max(0.0, min(float(img_w), min(x1, x2))), max(0.0, min(float(img_w), max(x1, x2)))
    y1, y2 = max(0.0, min(float(img_h), min(y1, y2))), max(0.0, min(float(img_h), max(y1, y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _fmt_coord(val):
    v = round(float(val), 1)
    return int(v) if v.is_integer() else v


def annotations_to_yolo_lines(annotations, img_w, img_h, class_names):
    """Serialize RoboLabel annotations to YOLO Detection (5 cols) or Segmentation (polygon) lines."""
    lines = []
    if img_w <= 0 or img_h <= 0:
        return lines

    for item in (annotations or []):
        cname = item.get("class_name")
        if cname not in class_names:
            continue
        c_idx = class_names.index(cname)
        box = _extract_clamped_bbox(item, img_w, img_h)
        if box is not None:
            x1, y1, x2, y2 = box
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            xc = ((x1 + x2) / 2.0) / img_w
            yc = ((y1 + y2) / 2.0) / img_h
            lines.append(f"{c_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        else:
            poly = item.get("polygon", [])
            if poly and len(poly) >= 3:
                coords = " ".join([
                    f"{max(0.0, min(1.0, float(pt[0]) / img_w)):.6f} {max(0.0, min(1.0, float(pt[1]) / img_h)):.6f}"
                    for pt in poly
                ])
                lines.append(f"{c_idx} {coords}")

    return lines


def annotations_to_labelme_json(annotations, img_name, img_w, img_h):
    """Serialize RoboLabel annotations to Labelme / X-AnyLabeling JSON preserving group_id."""
    shapes = []
    if img_w > 0 and img_h > 0:
        for item in (annotations or []):
            cname = str(item.get("class_name") or "").strip()
            if not cname:
                continue
            group_id = normalize_group_id(item.get("group_id"))
            box = _extract_clamped_bbox(item, img_w, img_h)
            if box is not None:
                x1, y1, x2, y2 = box
                shape_type = "rectangle"
                points = [[_fmt_coord(x1), _fmt_coord(y1)], [_fmt_coord(x2), _fmt_coord(y2)]]
            else:
                poly = item.get("polygon", [])
                if not poly or len(poly) < 3:
                    continue
                shape_type = "polygon"
                points = [
                    [
                        _fmt_coord(max(0.0, min(float(img_w), float(pt[0])))),
                        _fmt_coord(max(0.0, min(float(img_h), float(pt[1])))),
                    ]
                    for pt in poly
                ]
            shapes.append({
                "label": cname,
                "points": points,
                "group_id": group_id,
                "description": "",
                "shape_type": shape_type,
                "flags": {},
            })

    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(img_name),
        "imageData": None,
        "imageHeight": int(img_h),
        "imageWidth": int(img_w),
    }


def _build_yolo_yaml(root_dir, class_names, train_rel="images", val_rel="images", test_rel=""):
    norm_dir = root_dir.replace("\\", "/")
    names_block = "\n".join([f"  {idx}: {name}" for idx, name in enumerate(class_names)])
    test_line = f"test: {test_rel}\n" if test_rel else ""
    return f"path: {norm_dir}\ntrain: {train_rel}\nval: {val_rel}\n{test_line}\nnames:\n{names_block}\n"


def write_project_yaml_and_classes(project_dir, class_names):
    atomic_write_text(os.path.join(project_dir, "data.yaml"), _build_yolo_yaml(project_dir, class_names))
    atomic_write_text(os.path.join(project_dir, "classes.txt"), "\n".join(class_names) + "\n")


def _compute_project_stats(images_dir, annotations_dir):
    if not os.path.isdir(images_dir):
        return {"total_images": 0, "labeled_images": 0, "total_annotations": 0}
    imgs = [f for f in os.listdir(images_dir) if is_image_file(f)]
    labeled = 0
    total_anns = 0
    for f in imgs:
        ann_path = os.path.join(annotations_dir, f"{f}.json")
        if os.path.exists(ann_path):
            try:
                with open(ann_path, "r", encoding="utf-8") as af:
                    cnt = len(json.load(af).get("annotations", []))
                    if cnt > 0:
                        labeled += 1
                    total_anns += cnt
            except Exception:
                pass
    return {
        "total_images": len(imgs),
        "labeled_images": labeled,
        "total_annotations": total_anns,
    }


def _path_is_same_or_inside(candidate, other):
    c_abs = os.path.normcase(os.path.abspath(candidate))
    o_abs = os.path.normcase(os.path.abspath(other))
    if c_abs == o_abs:
        return True
    try:
        return os.path.commonpath([c_abs, o_abs]) == o_abs
    except ValueError:
        return False


class ProjectManager:
    """Manages active workspace/project state, safe dataset import, auto-save sync, and registry."""

    MANIFEST_FILENAME = "robolabel_project.json"
    DEFAULT_ID = "__default__"

    def __init__(self, base_dir):
        self.base_dir = os.path.abspath(base_dir)
        self.registry_path = os.path.join(self.base_dir, "projects_registry.json")
        self.default_projects_dir = os.path.join(self.base_dir, "projects")
        self.default_uploads = os.path.join(self.base_dir, "uploads")
        self.default_annotations = os.path.join(self.base_dir, "annotations")
        self.default_dataset = os.path.join(self.base_dir, "dataset")
        self.upload_sessions = {}

        os.makedirs(self.default_uploads, exist_ok=True)
        os.makedirs(self.default_annotations, exist_ok=True)
        os.makedirs(self.default_dataset, exist_ok=True)

        self.registry = self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("active_project_id", self.DEFAULT_ID)
                    data.setdefault("default_last_image", "")
                    data.setdefault("projects", [])
                    return data
            except Exception:
                pass
        return {
            "active_project_id": self.DEFAULT_ID,
            "default_last_image": "",
            "projects": [],
        }

    def _save_registry(self):
        atomic_write_json(self.registry_path, self.registry)

    def _find_registry_entry(self, project_id):
        for entry in self.registry.get("projects", []):
            if entry.get("id") == project_id:
                return entry
        return None

    def _upsert_registry_entry(self, manifest, stats=None):
        pid = manifest["id"]
        if stats is None:
            pdir = manifest["project_dir"]
            stats = _compute_project_stats(
                os.path.join(pdir, "images"),
                os.path.join(pdir, "annotations"),
            )
        record = {
            "id": pid,
            "name": manifest["name"],
            "project_dir": manifest["project_dir"],
            "dataset_format": manifest.get("dataset_format", "images_only"),
            "format_label": manifest.get("format_label", "Thư mục ảnh thông thường"),
            "classes": manifest.get("classes", list(DEFAULT_CLASS_NAMES)),
            "last_image": manifest.get("last_image", ""),
            "created_at": manifest.get("created_at", now_iso()),
            "updated_at": manifest.get("updated_at", now_iso()),
            "stats": stats,
        }
        projects = [p for p in self.registry.get("projects", []) if p.get("id") != pid and os.path.normcase(p.get("project_dir", "")) != os.path.normcase(manifest["project_dir"])]
        projects.insert(0, record)
        self.registry["projects"] = projects
        self._save_registry()
        return record

    def _read_manifest(self, project_dir):
        mpath = os.path.join(project_dir, self.MANIFEST_FILENAME)
        if not os.path.exists(mpath):
            return None
        with open(mpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["project_dir"] = os.path.abspath(project_dir)
        return data

    def _write_manifest(self, project_dir, manifest):
        mpath = os.path.join(project_dir, self.MANIFEST_FILENAME)
        atomic_write_json(mpath, manifest)

    def get_active_context(self):
        active_id = self.registry.get("active_project_id", self.DEFAULT_ID)
        if active_id != self.DEFAULT_ID:
            entry = self._find_registry_entry(active_id)
            if entry and os.path.isdir(entry.get("project_dir", "")):
                pdir = os.path.abspath(entry["project_dir"])
                manifest = self._read_manifest(pdir)
                if manifest:
                    images_dir = os.path.join(pdir, "images")
                    ann_dir = os.path.join(pdir, "annotations")
                    labels_dir = os.path.join(pdir, "labels")
                    dataset_dir = os.path.join(pdir, "dataset")
                    os.makedirs(images_dir, exist_ok=True)
                    os.makedirs(ann_dir, exist_ok=True)
                    os.makedirs(labels_dir, exist_ok=True)
                    os.makedirs(dataset_dir, exist_ok=True)
                    cnames, cinfo = build_class_info(manifest.get("classes"))
                    return {
                        "id": manifest["id"],
                        "name": manifest["name"],
                        "is_default": False,
                        "project_dir": pdir,
                        "images_dir": images_dir,
                        "annotations_dir": ann_dir,
                        "labels_dir": labels_dir,
                        "dataset_dir": dataset_dir,
                        "classes": cnames,
                        "class_info": cinfo,
                        "manifest": manifest,
                        "last_image": manifest.get("last_image", ""),
                        "dataset_format": manifest.get("dataset_format", "images_only"),
                        "format_label": manifest.get("format_label", "Thư mục ảnh thông thường"),
                    }
            # Fallback safely to default if missing on disk
            self.registry["active_project_id"] = self.DEFAULT_ID
            self._save_registry()

        return {
            "id": self.DEFAULT_ID,
            "name": "Workspace Mặc định",
            "is_default": True,
            "project_dir": self.base_dir,
            "images_dir": self.default_uploads,
            "annotations_dir": self.default_annotations,
            "labels_dir": self.default_annotations,
            "dataset_dir": self.default_dataset,
            "classes": list(DEFAULT_CLASS_NAMES),
            "class_info": [dict(c) for c in DEFAULT_CLASS_INFO],
            "manifest": None,
            "last_image": self.registry.get("default_last_image", ""),
            "dataset_format": "yolo_segmentation",
            "format_label": "Workspace Mặc định",
        }

    def list_projects_state(self):
        ctx = self.get_active_context()
        default_stats = _compute_project_stats(self.default_uploads, self.default_annotations)
        default_info = {
            "id": self.DEFAULT_ID,
            "name": "Workspace Mặc định",
            "project_dir": self.base_dir,
            "dataset_dir": self.default_dataset,
            "dataset_format": "default",
            "format_label": "Workspace Mặc định (uploads/ & annotations/)",
            "classes": list(DEFAULT_CLASS_NAMES),
            "last_image": self.registry.get("default_last_image", ""),
            "stats": default_stats,
            "is_default": True,
            "exists": True,
        }

        items = []
        changed = False
        for entry in self.registry.get("projects", []):
            pdir = entry.get("project_dir", "")
            exists = os.path.isdir(pdir) and os.path.exists(os.path.join(pdir, self.MANIFEST_FILENAME))
            item = dict(entry)
            item["dataset_dir"] = os.path.join(pdir, "dataset") if pdir else ""
            item["exists"] = exists
            item["is_default"] = False
            if exists:
                manifest = self._read_manifest(pdir)
                if manifest:
                    stats = _compute_project_stats(os.path.join(pdir, "images"), os.path.join(pdir, "annotations"))
                    item["name"] = manifest.get("name", item["name"])
                    item["classes"] = manifest.get("classes", item.get("classes", []))
                    item["last_image"] = manifest.get("last_image", "")
                    item["updated_at"] = manifest.get("updated_at", item.get("updated_at", ""))
                    item["dataset_format"] = manifest.get("dataset_format", item.get("dataset_format", "images_only"))
                    item["format_label"] = manifest.get("format_label", item.get("format_label", ""))
                    item["stats"] = stats
                    if stats != entry.get("stats"):
                        entry["stats"] = stats
                        changed = True
            items.append(item)

        if changed:
            self._save_registry()

        active_summary = default_info if ctx["is_default"] else next(
            (p for p in items if p["id"] == ctx["id"]),
            {
                "id": ctx["id"],
                "name": ctx["name"],
                "project_dir": ctx["project_dir"],
                "dataset_dir": ctx["dataset_dir"],
                "dataset_format": ctx["dataset_format"],
                "format_label": ctx["format_label"],
                "classes": ctx["classes"],
                "last_image": ctx["last_image"],
                "stats": _compute_project_stats(ctx["images_dir"], ctx["annotations_dir"]),
                "is_default": False,
                "exists": True,
            },
        )

        return {
            "active_project": active_summary,
            "default_workspace": default_info,
            "default_projects_dir": self.default_projects_dir,
            "projects": items,
        }

    def validate_new_project_target(self, save_dir, project_name, source_dir=None):
        clean_save_dir = (save_dir or "").strip() or self.default_projects_dir
        clean_name = (project_name or "").strip()
        if not clean_name:
            raise ValueError("Vui lòng nhập tên Project.")

        folder_name = sanitize_folder_name(clean_name)
        save_abs = os.path.abspath(clean_save_dir)
        project_dir = os.path.abspath(os.path.join(save_abs, folder_name))

        # Safety Rule 6: Never write into or overwrite the original source dataset!
        if source_dir:
            if _path_is_same_or_inside(project_dir, source_dir) or _path_is_same_or_inside(source_dir, project_dir):
                raise ValueError(
                    "Vị trí lưu không được trùng hoặc nằm bên trong thư mục dataset gốc để đảm bảo an toàn dữ liệu!"
                )

        # Also protect built-in system folders
        for protected in (
            self.default_uploads,
            self.default_annotations,
            self.default_dataset,
            os.path.join(self.base_dir, "static"),
            os.path.join(self.base_dir, "weights"),
            os.path.join(self.base_dir, "venv"),
        ):
            if _path_is_same_or_inside(project_dir, protected):
                raise ValueError("Không thể lưu project vào bên trong thư mục hệ thống của ứng dụng.")
        if os.path.normcase(project_dir) == os.path.normcase(self.base_dir):
            raise ValueError("Không thể dùng thư mục gốc ứng dụng làm thư mục project mới.")

        if os.path.exists(project_dir):
            if os.path.isfile(project_dir):
                raise ValueError(f"Đường dẫn '{project_dir}' đã tồn tại và là một tệp.")
            if os.listdir(project_dir):
                raise ValueError(
                    f"Thư mục '{project_dir}' đã tồn tại và không rỗng. Vui lòng đổi tên Project hoặc chọn vị trí lưu khác để tránh ghi đè dữ liệu."
                )

        return clean_name, folder_name, project_dir

    def _materialize_project(
        self,
        source_dir,
        project_dir,
        project_name,
        source_display_name="",
        source_path_record="",
        custom_classes=None,
        use_default_classes=False,
    ):
        scan = inspect_dataset_folder(source_dir)
        if scan["total_images"] == 0:
            raise ValueError("Không tìm thấy ảnh hợp lệ (.jpg, .jpeg, .png, .bmp, .webp) trong thư mục nguồn.")

        if custom_classes and len(custom_classes) > 0:
            chosen_names = [str(c).strip() for c in custom_classes if str(c).strip()]
        elif use_default_classes:
            chosen_names = list(DEFAULT_CLASS_NAMES)
        else:
            chosen_names = list(scan["detected_classes"])

        class_names, class_info = build_class_info(chosen_names)

        images_dir = os.path.join(project_dir, "images")
        ann_dir = os.path.join(project_dir, "annotations")
        labels_dir = os.path.join(project_dir, "labels")
        dataset_dir = os.path.join(project_dir, "dataset")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        splits_map = {}
        orig_rel_map = {}
        stored_files = []
        imported_objects = 0

        for entry in scan["entries"]:
            src_img = os.path.join(source_dir, entry["img_rel"])
            stored_name = entry["stored_name"]
            dst_img = os.path.join(images_dir, stored_name)

            try:
                with Image.open(src_img) as im:
                    img_w, img_h = im.size
            except Exception:
                continue

            # Copy source image read-only; never touch source
            shutil.copy2(src_img, dst_img)
            stored_files.append(stored_name)
            orig_rel_map[stored_name] = entry["img_rel"]
            if entry["split"]:
                splits_map[stored_name] = entry["split"]

            src_label_path = os.path.join(source_dir, entry["label_rel"]) if entry["label_rel"] else None
            anns = parse_yolo_label_file_to_annotations(src_label_path, img_w, img_h, class_names)
            imported_objects += len(anns)

            # Save canonical JSON atomically
            atomic_write_json(os.path.join(ann_dir, f"{stored_name}.json"), {"annotations": anns})

            # Save synced YOLO .txt atomically
            stem = os.path.splitext(stored_name)[0]
            yolo_lines = annotations_to_yolo_lines(anns, img_w, img_h, class_names)
            atomic_write_text(os.path.join(labels_dir, f"{stem}.txt"), "\n".join(yolo_lines))

        if not stored_files:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise ValueError("Không đọc được ảnh hợp lệ nào từ thư mục nguồn.")

        # Re-normalize class_info if label conversion expanded classes dynamically
        class_names, class_info = build_class_info(class_names)
        write_project_yaml_and_classes(project_dir, class_names)

        timestamp = now_iso()
        project_id = f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        manifest = {
            "version": 1,
            "id": project_id,
            "name": project_name,
            "project_dir": project_dir,
            "dataset_format": scan["dataset_format"],
            "format_label": scan["format_label"],
            "classes": class_names,
            "class_info": class_info,
            "splits": splits_map,
            "original_rel_paths": orig_rel_map,
            "last_image": stored_files[0],
            "source_info": {
                "source_name": source_display_name or os.path.basename(source_dir),
                "source_path": source_path_record,
                "classes_source": scan["classes_source"],
                "imported_images": len(stored_files),
                "imported_objects": imported_objects,
            },
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        self._write_manifest(project_dir, manifest)

        stats = _compute_project_stats(images_dir, ann_dir)
        record = self._upsert_registry_entry(manifest, stats=stats)
        self.registry["active_project_id"] = project_id
        self._save_registry()

        return {
            "project": record,
            "scan": {
                "dataset_format": scan["dataset_format"],
                "format_label": scan["format_label"],
                "imported_images": len(stored_files),
                "imported_objects": imported_objects,
                "detection_objects": scan["detection_objects"],
                "segmentation_objects": scan["segmentation_objects"],
                "classes": class_names,
            },
        }

    def create_project_from_local_folder(
        self,
        source_dir,
        save_dir,
        project_name,
        custom_classes=None,
        use_default_classes=False,
    ):
        if not source_dir or not os.path.isdir(source_dir):
            raise ValueError("Thư mục dataset nguồn không tồn tại.")
        source_abs = os.path.abspath(source_dir)
        clean_name, _, project_dir = self.validate_new_project_target(save_dir, project_name, source_dir=source_abs)
        os.makedirs(project_dir, exist_ok=True)
        try:
            return self._materialize_project(
                source_dir=source_abs,
                project_dir=project_dir,
                project_name=clean_name,
                source_display_name=os.path.basename(source_abs),
                source_path_record=source_abs,
                custom_classes=custom_classes,
                use_default_classes=use_default_classes,
            )
        except Exception:
            if os.path.isdir(project_dir):
                shutil.rmtree(project_dir, ignore_errors=True)
            raise

    def init_upload_session(self, save_dir, project_name, source_name=""):
        clean_name, _, project_dir = self.validate_new_project_target(save_dir, project_name, source_dir=None)
        session_id = uuid.uuid4().hex
        staging_dir = os.path.join(project_dir, "_import_tmp")
        os.makedirs(staging_dir, exist_ok=True)
        self.upload_sessions[session_id] = {
            "session_id": session_id,
            "project_name": clean_name,
            "project_dir": project_dir,
            "staging_dir": staging_dir,
            "source_name": source_name or clean_name,
            "file_count": 0,
        }
        return {
            "session_id": session_id,
            "project_dir": project_dir,
            "project_name": clean_name,
        }

    def add_upload_chunk(self, session_id, file_items):
        sess = self.upload_sessions.get(session_id)
        if not sess:
            raise ValueError("Phiên nhập dữ liệu không tồn tại hoặc đã hết hạn.")

        staging_abs = os.path.abspath(sess["staging_dir"])
        saved = 0
        for raw_rel, file_obj in file_items:
            norm = (raw_rel or getattr(file_obj, "filename", "") or "").replace("\\", "/").lstrip("/")
            if not norm:
                continue
            parts = [p for p in norm.split("/") if p not in ("", ".")]
            if not parts or any(p == ".." for p in parts):
                continue
            ext = os.path.splitext(parts[-1])[1].lower()
            if ext not in ALLOWED_IMPORT_EXTS:
                continue
            dst_path = os.path.abspath(os.path.join(staging_abs, *parts))
            if not _path_is_same_or_inside(dst_path, staging_abs):
                continue
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            file_obj.save(dst_path)
            saved += 1
            sess["file_count"] += 1
        return {"saved": saved, "total_received": sess["file_count"]}

    def finalize_upload_session(self, session_id, custom_classes=None, use_default_classes=False):
        sess = self.upload_sessions.pop(session_id, None)
        if not sess:
            raise ValueError("Phiên nhập dữ liệu không tồn tại hoặc đã kết thúc.")

        project_dir = sess["project_dir"]
        staging_dir = sess["staging_dir"]
        try:
            result = self._materialize_project(
                source_dir=staging_dir,
                project_dir=project_dir,
                project_name=sess["project_name"],
                source_display_name=sess["source_name"],
                source_path_record="Trình duyệt (Kéo thả / Chọn thư mục)",
                custom_classes=custom_classes,
                use_default_classes=use_default_classes,
            )
            shutil.rmtree(staging_dir, ignore_errors=True)
            return result
        except Exception:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise

    def cancel_upload_session(self, session_id):
        sess = self.upload_sessions.pop(session_id, None)
        if sess and os.path.isdir(sess["project_dir"]):
            shutil.rmtree(sess["project_dir"], ignore_errors=True)
        return True

    def switch_project(self, project_id=None, project_dir=None):
        if project_id == self.DEFAULT_ID:
            self.registry["active_project_id"] = self.DEFAULT_ID
            self._save_registry()
            return self.list_projects_state()

        if project_dir:
            return self.open_existing_project_folder(project_dir)

        entry = self._find_registry_entry(project_id)
        if not entry:
            raise ValueError("Không tìm thấy project đã chọn trong danh sách.")

        pdir = entry.get("project_dir", "")
        if not os.path.isdir(pdir) or not os.path.exists(os.path.join(pdir, self.MANIFEST_FILENAME)):
            raise ValueError(f"Thư mục project '{pdir}' không còn tồn tại hoặc bị thiếu file robolabel_project.json.")

        self.registry["active_project_id"] = project_id
        self._save_registry()
        return self.list_projects_state()

    def open_existing_project_folder(self, project_dir):
        if not project_dir or not os.path.isdir(project_dir):
            raise ValueError("Thư mục project không tồn tại.")
        pdir = os.path.abspath(project_dir)
        manifest = self._read_manifest(pdir)
        if not manifest:
            raise ValueError(
                "Thư mục này chưa phải Project RoboLabel (thiếu robolabel_project.json). "
                "Nếu đây là thư mục ảnh hoặc dataset YOLO gốc, hãy dùng 'Tạo Project Mới' để nhập an toàn mà không ghi đè dữ liệu gốc."
            )

        manifest.setdefault("id", f"proj_{uuid.uuid4().hex[:10]}")
        manifest["project_dir"] = pdir
        self._write_manifest(pdir, manifest)
        self._upsert_registry_entry(manifest)
        self.registry["active_project_id"] = manifest["id"]
        self._save_registry()
        return self.list_projects_state()

    def remove_from_registry(self, project_id):
        if project_id == self.DEFAULT_ID:
            raise ValueError("Không thể gỡ Workspace Mặc định.")
        self.registry["projects"] = [p for p in self.registry.get("projects", []) if p.get("id") != project_id]
        if self.registry.get("active_project_id") == project_id:
            self.registry["active_project_id"] = self.DEFAULT_ID
        self._save_registry()
        return self.list_projects_state()

    def sync_image_annotations(self, filename, annotations_list, update_last_image=True):
        ctx = self.get_active_context()
        safe_name = os.path.basename(filename)
        if safe_name != filename:
            raise ValueError("Tên ảnh không hợp lệ.")

        img_path = os.path.join(ctx["images_dir"], safe_name)
        ann_path = os.path.join(ctx["annotations_dir"], f"{safe_name}.json")
        if not isinstance(annotations_list, list):
            annotations_list = []
        annotations_list = sanitize_annotations_list(annotations_list)

        # Atomic JSON write (canonical source of truth)
        atomic_write_json(ann_path, {"annotations": annotations_list})

        # Sync YOLO .txt immediately to prevent label drift / data loss
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
                yolo_lines = annotations_to_yolo_lines(annotations_list, img_w, img_h, ctx["classes"])
                stem = os.path.splitext(safe_name)[0]
                txt_path = os.path.join(ctx["labels_dir"], f"{stem}.txt")
                atomic_write_text(txt_path, "\n".join(yolo_lines))
            except Exception as e:
                print(f"[!] Warning syncing YOLO label for {safe_name}: {e}")

        if ctx["is_default"]:
            if update_last_image:
                self.registry["default_last_image"] = safe_name
                self._save_registry()
        else:
            manifest = ctx["manifest"]
            if manifest:
                manifest["updated_at"] = now_iso()
                if update_last_image:
                    manifest["last_image"] = safe_name
                self._write_manifest(ctx["project_dir"], manifest)
                self._upsert_registry_entry(manifest)

        return {"success": True, "saved_at": now_iso(), "count": len(annotations_list)}

    def record_last_image(self, filename):
        ctx = self.get_active_context()
        safe_name = os.path.basename(filename)
        if not os.path.exists(os.path.join(ctx["images_dir"], safe_name)):
            return
        if ctx["is_default"]:
            if self.registry.get("default_last_image") != safe_name:
                self.registry["default_last_image"] = safe_name
                self._save_registry()
        else:
            manifest = ctx["manifest"]
            if manifest and manifest.get("last_image") != safe_name:
                manifest["last_image"] = safe_name
                self._write_manifest(ctx["project_dir"], manifest)

    def _resolve_export_dir(self, ctx, output_dir):
        target = os.path.abspath((output_dir or "").strip() or ctx["dataset_dir"])
        protected = (
            ctx["images_dir"],
            ctx["annotations_dir"],
            self.default_uploads,
            self.default_annotations,
            os.path.join(self.base_dir, "static"),
            os.path.join(self.base_dir, "weights"),
            os.path.join(self.base_dir, "venv"),
            os.path.join(self.base_dir, ".git"),
        )
        if os.path.normcase(target) == os.path.normcase(self.base_dir):
            raise ValueError("Không thể xuất đè trực tiếp vào thư mục gốc ứng dụng.")
        if not ctx["is_default"] and os.path.normcase(target) == os.path.normcase(ctx["project_dir"]):
            raise ValueError("Không thể xuất đè trực tiếp lên thư mục gốc của Project.")
        for p in protected:
            if _path_is_same_or_inside(target, p):
                raise ValueError("Thư mục xuất không được trùng hoặc nằm bên trong thư mục ảnh gốc / hệ thống.")
        os.makedirs(target, exist_ok=True)
        return target

    def _resolve_export_classes(self, ctx, class_names):
        if class_names is None:
            return list(ctx["classes"])
        if isinstance(class_names, str):
            raw = [part.strip() for part in class_names.split(",")]
        elif isinstance(class_names, list):
            raw = [str(part).strip() for part in class_names]
        else:
            raise ValueError("Danh sách class xuất không hợp lệ.")
        clean = []
        seen = set()
        for name in raw:
            if name and name not in seen:
                clean.append(name)
                seen.add(name)
        if not clean:
            raise ValueError("Danh sách class xuất không được rỗng.")
        return clean

    def _read_image_annotations(self, ann_dir, img_name):
        ann_path = os.path.join(ann_dir, f"{img_name}.json")
        if not os.path.exists(ann_path):
            return []
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                return json.load(f).get("annotations", [])
        except Exception as e:
            print(f"[!] Error reading {ann_path}: {e}")
            return []

    def export_single_image(self, image_name, export_format="both", class_names=None):
        ctx = self.get_active_context()
        fmt = (export_format or "both").strip().lower()
        if fmt not in ("txt", "json", "both"):
            raise ValueError("Định dạng xuất không hợp lệ (chỉ chấp nhận 'txt', 'json', 'both').")
        safe_name = os.path.basename((image_name or "").strip())
        if not safe_name or safe_name != (image_name or "").strip():
            raise ValueError("Tên ảnh không hợp lệ.")
        img_path = os.path.join(ctx["images_dir"], safe_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {safe_name}")

        export_classes = self._resolve_export_classes(ctx, class_names)
        with Image.open(img_path) as im:
            w, h = im.size
        ann_items = self._read_image_annotations(ctx["annotations_dir"], safe_name)
        stem = os.path.splitext(safe_name)[0]

        txt_content = "\n".join(annotations_to_yolo_lines(ann_items, w, h, export_classes))
        json_obj = annotations_to_labelme_json(ann_items, safe_name, w, h)
        return {
            "format": fmt,
            "image_name": safe_name,
            "stem": stem,
            "txt_content": txt_content,
            "json_content": json.dumps(json_obj, ensure_ascii=False, indent=2),
            "classes_txt": "\n".join(export_classes) + "\n",
            "classes": export_classes,
            "count": len(ann_items),
        }

    def export_dataset(self, export_format="txt", layout="split", output_dir=None, class_names=None):
        ctx = self.get_active_context()
        fmt = (export_format or "txt").strip().lower()
        if fmt not in ("txt", "json", "both"):
            raise ValueError("Định dạng xuất không hợp lệ (chỉ chấp nhận 'txt', 'json', 'both').")
        lay = (layout or "split").strip().lower()
        if lay not in ("split", "flat"):
            raise ValueError("Cấu trúc thư mục xuất không hợp lệ (chỉ chấp nhận 'split' hoặc 'flat').")

        images_dir = ctx["images_dir"]
        ann_dir = ctx["annotations_dir"]
        labels_dir = ctx["labels_dir"]
        export_classes = self._resolve_export_classes(ctx, class_names)

        images = sorted([f for f in os.listdir(images_dir) if is_image_file(f)])
        if not images:
            raise ValueError("Không có ảnh nào trong project hiện tại để xuất.")

        target_dir = self._resolve_export_dir(ctx, output_dir)
        want_txt = fmt in ("txt", "both")
        want_json = fmt in ("json", "both")
        is_default_target = os.path.normcase(target_dir) == os.path.normcase(ctx["dataset_dir"])
        sync_project_labels = want_txt and is_default_target and (export_classes == ctx["classes"])

        if lay == "split":
            for split in ("train", "val", "test"):
                for sub in ("images", "labels", "annotations_json"):
                    target_sub = os.path.join(target_dir, sub, split)
                    if os.path.isdir(target_sub):
                        shutil.rmtree(target_sub, ignore_errors=True)
                    if split in ("train", "val") and (
                        sub == "images" or (sub == "labels" and want_txt) or (sub == "annotations_json" and want_json)
                    ):
                        os.makedirs(target_sub, exist_ok=True)

            recorded_splits = (ctx["manifest"] or {}).get("splits", {})
            has_explicit_train_val = (
                any(recorded_splits.get(img) == "train" for img in images)
                and any(recorded_splits.get(img) == "val" for img in images)
            )
            if has_explicit_train_val:
                split_for = {img: (recorded_splits.get(img) or "train") for img in images}
            else:
                val_ratio = 0.2
                shuffled = list(images)
                random.seed(42)
                random.shuffle(shuffled)
                val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
                val_set = set(shuffled[:val_count])
                split_for = {img: ("val" if img in val_set else "train") for img in images}
        else:
            split_for = {img: "" for img in images}

        stats = {
            "total_images": len(images),
            "train_images": 0,
            "val_images": 0,
            "test_images": 0,
            "total_annotations": 0,
            "grouped_annotations": 0,
            "class_counts": {c: 0 for c in export_classes},
        }

        for img_name in images:
            src_img = os.path.join(images_dir, img_name)
            stem = os.path.splitext(img_name)[0]
            with Image.open(src_img) as im:
                w, h = im.size

            ann_items = self._read_image_annotations(ann_dir, img_name)
            for item in ann_items:
                cname = (item.get("class_name") or "").strip()
                counted = False
                if cname in stats["class_counts"]:
                    stats["class_counts"][cname] += 1
                    stats["total_annotations"] += 1
                    counted = True
                elif want_json and cname:
                    stats["total_annotations"] += 1
                    counted = True
                if counted and normalize_group_id(item.get("group_id")) is not None:
                    stats["grouped_annotations"] += 1

            if lay == "split":
                split = split_for[img_name]
                if split not in ("train", "val", "test"):
                    split = "train"
                os.makedirs(os.path.join(target_dir, "images", split), exist_ok=True)
                shutil.copy2(src_img, os.path.join(target_dir, "images", split, img_name))
                stats[f"{split}_images"] += 1

                if want_txt:
                    os.makedirs(os.path.join(target_dir, "labels", split), exist_ok=True)
                    txt_content = "\n".join(annotations_to_yolo_lines(ann_items, w, h, export_classes))
                    atomic_write_text(os.path.join(target_dir, "labels", split, f"{stem}.txt"), txt_content)
                    if sync_project_labels:
                        atomic_write_text(os.path.join(labels_dir, f"{stem}.txt"), txt_content)
                if want_json:
                    os.makedirs(os.path.join(target_dir, "annotations_json", split), exist_ok=True)
                    json_obj = annotations_to_labelme_json(ann_items, img_name, w, h)
                    atomic_write_json(os.path.join(target_dir, "annotations_json", split, f"{stem}.json"), json_obj)
            else:
                shutil.copy2(src_img, os.path.join(target_dir, img_name))
                if want_txt:
                    txt_content = "\n".join(annotations_to_yolo_lines(ann_items, w, h, export_classes))
                    atomic_write_text(os.path.join(target_dir, f"{stem}.txt"), txt_content)
                    if sync_project_labels:
                        atomic_write_text(os.path.join(labels_dir, f"{stem}.txt"), txt_content)
                if want_json:
                    json_obj = annotations_to_labelme_json(ann_items, img_name, w, h)
                    atomic_write_json(os.path.join(target_dir, f"{stem}.json"), json_obj)

        yaml_path = None
        if want_txt:
            if lay == "split":
                yaml_content = _build_yolo_yaml(
                    target_dir,
                    export_classes,
                    train_rel="images/train",
                    val_rel="images/val",
                    test_rel="images/test" if stats["test_images"] > 0 else "",
                )
            else:
                yaml_content = _build_yolo_yaml(target_dir, export_classes, train_rel=".", val_rel=".")
            yaml_path = os.path.join(target_dir, "data.yaml")
            atomic_write_text(yaml_path, yaml_content)
            atomic_write_text(os.path.join(target_dir, "classes.txt"), "\n".join(export_classes) + "\n")

        fmt_desc = {"txt": "YOLO (.txt)", "json": "JSON (Labelme + Group ID)", "both": "YOLO (.txt) + JSON"}[fmt]
        return {
            "success": True,
            "message": f"Xuất nhãn {fmt_desc} thành công! Dữ liệu đã lưu tại:\n{target_dir}",
            "format": fmt,
            "layout": lay,
            "dataset_dir": target_dir,
            "data_yaml": yaml_path,
            "json_dir": (os.path.join(target_dir, "annotations_json") if lay == "split" else target_dir) if want_json else None,
            "stats": stats,
            "classes": export_classes,
            "project_name": ctx["name"],
        }

    def export_active_yolo(self):
        return self.export_dataset(export_format="txt", layout="split")
