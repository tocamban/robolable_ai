"""Self-check covering device_manager, project_manager, Flask routes, and optional live SAM+CLIP smoke."""

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

from PIL import Image, ImageDraw
from shapely.geometry import Polygon as ShapelyPolygon

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import app as robolabel_app
import device_manager as dm
import project_manager as pm_mod


def _make_sample_dataset(src_dir):
    os.makedirs(os.path.join(src_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(src_dir, "labels", "train"), exist_ok=True)
    img_path = os.path.join(src_dir, "images", "train", "sample.jpg")
    im = Image.new("RGB", (120, 100), color=(220, 230, 210))
    ImageDraw.Draw(im).ellipse([30, 20, 90, 80], fill=(205, 35, 35))
    im.save(img_path)
    with open(os.path.join(src_dir, "labels", "train", "sample.txt"), "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.5 0.6\n")
    with open(os.path.join(src_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("ripe\ncalyx\n")


def check_device_manager(tmp):
    wdir = os.path.join(tmp, "weights")
    bdir = os.path.join(tmp, "base")
    os.makedirs(wdir, exist_ok=True)
    os.makedirs(bdir, exist_ok=True)

    root_w = os.path.join(bdir, "m.pt")
    sub_w = os.path.join(wdir, "m.pt")
    open(root_w, "wb").write(b"1")
    assert dm.resolve_weight_path(wdir, bdir, "m.pt") == root_w
    open(sub_w, "wb").write(b"2")
    assert dm.resolve_weight_path(wdir, bdir, "m.pt") == sub_w
    assert dm.resolve_weight_path(wdir, bdir, "missing.pt") is None
    assert dm.resolve_weight_path(wdir, bdir, "missing.pt", fallback_to_weights=True) == os.path.join(wdir, "missing.pt")

    dm.save_device_preference(bdir, "cpu")
    assert dm.load_device_preference(bdir) == "cpu"
    dm.save_device_preference(bdir, "auto")
    assert dm.load_device_preference(bdir) == "auto"
    try:
        dm.save_device_preference(bdir, "tpu")
        assert False, "expected ValueError for invalid device"
    except ValueError:
        pass

    assert dm.resolve_runtime_device("cpu")[0] == "cpu"

    suit = dm.evaluate_gpu_suitability(
        {"gpu_detected": False, "cuda_available": False},
        {"sam_tiny": {"avg_ms": 1200.0}, "tensor": {"avg_ms": 10.0}},
        {"executed": False},
        wdir,
        bdir,
    )
    assert suit["overall_status"] == "cpu_only"
    assert [m["id"] for m in suit["model_compatibility"]] == ["sam2_tiny", "clip_vit_b32", "sam2_large"]


def check_project_manager(tmp):
    assert pm_mod._find_matching_label(
        "images/train/a.jpg",
        {"labels/train/a.txt"},
        {"labels/train/a.txt": "labels/train/a.txt"},
        {"a": ["labels/train/a.txt"]},
    ) == "labels/train/a.txt"

    src_dir = os.path.join(tmp, "src_ds")
    save_dir = os.path.join(tmp, "saved_projects")
    base_dir = os.path.join(tmp, "app_base")
    _make_sample_dataset(src_dir)

    mgr = pm_mod.ProjectManager(base_dir)
    try:
        mgr.validate_new_project_target(src_dir, "nested_bad", source_dir=src_dir)
        assert False, "expected ValueError when target is inside source"
    except ValueError:
        pass

    created = mgr.create_project_from_local_folder(src_dir, save_dir, "Self Check Proj")
    pid = created["project"]["id"]
    pdir = created["project"]["project_dir"]
    assert os.path.exists(os.path.join(pdir, "data.yaml"))
    assert os.path.exists(os.path.join(pdir, "classes.txt"))

    # Group ID normalization assertions
    assert pm_mod.normalize_group_id(1) == 1
    assert pm_mod.normalize_group_id("2") == 2
    assert pm_mod.normalize_group_id(0) == 0
    assert pm_mod.normalize_group_id("") is None
    assert pm_mod.normalize_group_id(None) is None
    assert pm_mod.normalize_group_id(True) is None
    assert pm_mod.normalize_group_id(-1) is None
    assert pm_mod.normalize_group_id("-3") is None
    sanitized_label = pm_mod.sanitize_annotations_list([{"label": "ripe", "group_id": "2", "polygon": [[0, 0], [10, 0], [10, 10]]}, "bad"])
    assert len(sanitized_label) == 1 and sanitized_label[0]["class_name"] == "ripe" and sanitized_label[0]["group_id"] == 2

    sample_anns = [
        {"class_name": "ripe", "group_id": 1, "type": "polygon", "polygon": [[12, 10], [60, 10], [60, 50], [12, 50]]},
        {"class_name": "calyx", "group_id": "1", "type": "polygon", "polygon": [[12, 5], [60, 5], [60, 10], [12, 10]]},
        # Split stem into two polygons sharing group_id=1
        {"class_name": "stem", "group_id": 1, "type": "polygon", "polygon": [[30, 1], [36, 1], [36, 5], [30, 5]]},
        {"class_name": "stem", "group_id": 1, "type": "polygon", "polygon": [[40, 1], [48, 1], [48, 4], [40, 4]]},
        # Bbox with group_id=0 (must preserve falsy 0) + ungrouped polygon
        {"class_name": "ripe", "group_id": 0, "type": "bbox", "bbox": [70, 20, 100, 60]},
        {"class_name": "unripe", "group_id": None, "type": "polygon", "polygon": [[72, 65], [96, 65], [96, 90], [72, 90]]},
    ]
    synced = mgr.sync_image_annotations("sample.jpg", sample_anns)
    assert synced["success"] is True and synced["count"] == 6

    with open(os.path.join(pdir, "annotations", "sample.jpg.json"), "r", encoding="utf-8") as f:
        saved_items = json.load(f)["annotations"]
    assert len(saved_items) == 6
    assert saved_items[0]["group_id"] == 1
    assert saved_items[1]["group_id"] == 1
    assert saved_items[4]["group_id"] == 0
    assert "group_id" not in saved_items[5]

    exported = mgr.export_active_yolo()
    assert exported["success"] is True
    assert os.path.exists(exported["data_yaml"])

    # Export both (.txt YOLO + .json Labelme) in flat layout with custom 5-class order
    custom_order = ["unripe", "almost_ripe", "ripe", "stem", "calyx"]
    flat_out = os.path.join(tmp, "export_flat")
    exp_flat = mgr.export_dataset(
        export_format="both",
        layout="flat",
        output_dir=flat_out,
        class_names=custom_order,
    )
    assert exp_flat["success"] is True and exp_flat["format"] == "both" and exp_flat["layout"] == "flat"
    assert exp_flat["stats"]["grouped_annotations"] == 5
    assert exp_flat["classes"] == custom_order

    flat_json_path = os.path.join(flat_out, "sample.json")
    flat_txt_path = os.path.join(flat_out, "sample.txt")
    assert os.path.exists(flat_json_path) and os.path.exists(flat_txt_path)
    with open(flat_json_path, "r", encoding="utf-8") as f:
        lm_data = json.load(f)
    assert lm_data["imagePath"] == "sample.jpg"
    assert lm_data["imageWidth"] == 120 and lm_data["imageHeight"] == 100
    assert len(lm_data["shapes"]) == 6
    assert lm_data["shapes"][2]["label"] == "stem" and lm_data["shapes"][2]["group_id"] == 1
    assert lm_data["shapes"][3]["label"] == "stem" and lm_data["shapes"][3]["group_id"] == 1
    assert lm_data["shapes"][4]["shape_type"] == "rectangle" and lm_data["shapes"][4]["group_id"] == 0
    assert lm_data["shapes"][5]["group_id"] is None

    with open(flat_txt_path, "r", encoding="utf-8") as f:
        txt_lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    assert len(txt_lines) == 6
    assert "group_id" not in "\n".join(txt_lines)
    # 0:unripe, 1:almost_ripe, 2:ripe, 3:stem, 4:calyx
    assert txt_lines[0].startswith("2 0.100000 0.100000 ")
    assert txt_lines[1].startswith("4 ")
    assert txt_lines[2].startswith("3 ") and txt_lines[3].startswith("3 ")
    assert txt_lines[4].startswith("2 ")
    assert txt_lines[5].startswith("0 ")

    # Split layout into custom dir with both formats
    split_out = os.path.join(tmp, "export_split")
    exp_split = mgr.export_dataset(
        export_format="both",
        layout="split",
        output_dir=split_out,
        class_names=custom_order,
    )
    assert exp_split["success"] is True and os.path.exists(exp_split["data_yaml"])
    assert os.path.exists(os.path.join(split_out, "labels", "train", "sample.txt"))
    assert os.path.exists(os.path.join(split_out, "annotations_json", "train", "sample.json"))

    # Subset-class export consistency: grouped_annotations never exceeds total_annotations
    subset_out = os.path.join(tmp, "export_subset")
    exp_sub = mgr.export_dataset(export_format="both", layout="flat", output_dir=subset_out, class_names=["ripe"])
    assert exp_sub["stats"]["total_annotations"] == 6
    assert exp_sub["stats"]["grouped_annotations"] == 5
    exp_sub_txt = mgr.export_dataset(export_format="txt", layout="flat", output_dir=subset_out, class_names=["ripe"])
    assert exp_sub_txt["stats"]["total_annotations"] == 2
    assert exp_sub_txt["stats"]["grouped_annotations"] == 2

    # Protected directory guard
    for bad_dir in (pdir, os.path.join(pdir, "images")):
        try:
            mgr.export_dataset(export_format="json", layout="flat", output_dir=bad_dir)
            assert False, f"expected ValueError for protected output_dir {bad_dir}"
        except ValueError:
            pass

    mgr.remove_from_registry(pid)
    reopened = mgr.open_existing_project_folder(pdir)
    assert reopened["active_project"]["id"] == pid

    # Delete image from project: removes image, .json, .txt, and dataset export copies + updates manifest
    assert os.path.exists(os.path.join(pdir, "images", "sample.jpg"))
    assert os.path.exists(os.path.join(pdir, "annotations", "sample.jpg.json"))
    assert os.path.exists(os.path.join(pdir, "labels", "sample.txt"))
    assert os.path.exists(os.path.join(pdir, "dataset", "images", "train", "sample.jpg"))
    assert os.path.exists(os.path.join(pdir, "dataset", "labels", "train", "sample.txt"))
    del_proj_res = mgr.delete_image("sample.jpg")
    assert del_proj_res["success"] is True and del_proj_res["remaining_images"] == 0
    assert not os.path.exists(os.path.join(pdir, "images", "sample.jpg"))
    assert not os.path.exists(os.path.join(pdir, "annotations", "sample.jpg.json"))
    assert not os.path.exists(os.path.join(pdir, "labels", "sample.txt"))
    assert not os.path.exists(os.path.join(pdir, "dataset", "images", "train", "sample.jpg"))
    assert not os.path.exists(os.path.join(pdir, "dataset", "labels", "train", "sample.txt"))
    ctx_after_del = mgr.get_active_context()
    assert "sample.jpg" not in (ctx_after_del["manifest"].get("splits") or {})
    assert ctx_after_del["manifest"].get("last_image") == ""

    rest_proj_res = mgr.restore_deleted_image()
    assert rest_proj_res["success"] is True and rest_proj_res["restored_image"] == "sample.jpg"
    assert os.path.exists(os.path.join(pdir, "images", "sample.jpg"))
    assert os.path.exists(os.path.join(pdir, "annotations", "sample.jpg.json"))
    assert os.path.exists(os.path.join(pdir, "labels", "sample.txt"))
    assert os.path.exists(os.path.join(pdir, "dataset", "images", "train", "sample.jpg"))
    assert os.path.exists(os.path.join(pdir, "dataset", "labels", "train", "sample.txt"))
    ctx_after_rest = mgr.get_active_context()
    assert ctx_after_rest["manifest"].get("splits", {}).get("sample.jpg") == "train"
    assert ctx_after_rest["manifest"].get("last_image") == "sample.jpg"

    # Corrupt image during export_dataset must fail BEFORE wiping dataset split directories
    corrupt_img = os.path.join(pdir, "images", "corrupt.jpg")
    with open(corrupt_img, "wb") as cf:
        cf.write(b"not-an-image")
    try:
        try:
            mgr.export_dataset(export_format="txt", layout="split")
            raise AssertionError("Expected ValueError on corrupt image export")
        except ValueError as exc:
            assert "corrupt.jpg" in str(exc)
        assert os.path.exists(os.path.join(pdir, "dataset", "images", "train", "sample.jpg"))
    finally:
        if os.path.exists(corrupt_img):
            os.remove(corrupt_img)

    del_proj_again = mgr.delete_image("sample.jpg")
    assert del_proj_again["success"] is True and del_proj_again["remaining_images"] == 0
    assert not os.path.exists(os.path.join(pdir, "images", "sample.jpg"))


def _img_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (80, 60), color=(40, 160, 40)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def check_flask_routes(tmp):
    src_dir = os.path.join(tmp, "api_src")
    save_dir = os.path.join(tmp, "api_save")
    base_dir = os.path.join(tmp, "api_base")
    _make_sample_dataset(src_dir)

    orig_pm = robolabel_app.pm
    robolabel_app.pm = pm_mod.ProjectManager(base_dir)
    client = robolabel_app.app.test_client()
    try:
        index_resp = client.get("/")
        assert index_resp.status_code == 200 and b"labelScale" in index_resp.data and b"IntersectionObserver" in index_resp.data
        assert client.get("/api/system_info").status_code == 200
        assert len(client.get("/api/models").get_json()) >= 2
        assert "classes" in client.get("/api/classes").get_json()
        assert client.get("/api/images").status_code == 200
        assert client.get("/api/projects").status_code == 200

        # Upload, image fetch, thumbnail fetch, annotation save/load & path traversal check
        assert client.post("/api/upload", data={}).status_code == 400
        assert client.post("/api/upload", data={"file": (io.BytesIO(b"bad"), "evil.txt")}).status_code == 400
        up = client.post("/api/upload", data={"file": (_img_bytes(), "up_test.jpg")})
        up_json = up.get_json()
        assert up.status_code == 200 and up_json["filename"] == "up_test.jpg" and "thumb_url" in up_json
        img_resp = client.get("/api/image/up_test.jpg")
        assert img_resp.status_code == 200 and "no-store" in img_resp.headers.get("Cache-Control", "")
        img_resp.close()
        thumb_resp = client.get("/api/thumb/up_test.jpg")
        assert thumb_resp.status_code == 200 and thumb_resp.mimetype == "image/jpeg"
        assert "private" in thumb_resp.headers.get("Cache-Control", "")
        with Image.open(io.BytesIO(thumb_resp.data)) as th_im:
            assert th_im.size[0] <= 160 and th_im.size[1] <= 140
        thumb_resp.close()
        assert client.get("/api/thumb/sub/bad.jpg").status_code == 400
        assert client.get("/api/thumb/bad.txt").status_code == 400
        assert client.get("/api/thumb/missing.jpg").status_code == 404
        up_ctx = robolabel_app.pm.get_active_context()
        disk_img_path = os.path.join(up_ctx["images_dir"], "disk_only.jpg")
        disk_json_path = os.path.join(up_ctx["annotations_dir"], "disk_only.jpg.json")
        with open(disk_img_path, "wb") as df:
            df.write(_img_bytes().getvalue())
        try:
            imgs_payload = client.get("/api/images").get_json()
            assert any(it["filename"] == "up_test.jpg" and "/api/thumb/" in it.get("thumb_url", "") for it in imgs_payload)
            assert any(it["filename"] == "disk_only.jpg" and it["annotations_count"] == 0 for it in imgs_payload)
            assert not os.path.exists(disk_json_path), "GET /api/images must not create empty .json files"
            with open(disk_json_path, "w", encoding="utf-8") as cf:
                cf.write("{corrupt")
            corrupt_ann = client.get("/api/annotations/disk_only.jpg")
            corrupt_json = corrupt_ann.get_json()
            assert corrupt_ann.status_code == 200 and corrupt_json["annotations"] == [] and corrupt_json["has_overlap"] is False
        finally:
            for fp in (disk_img_path, disk_json_path):
                if os.path.exists(fp):
                    os.remove(fp)
        assert client.get("/api/annotations/sub/bad.jpg").status_code == 400

        save_ann = client.post(
            "/api/annotations/up_test.jpg",
            json={"annotations": [
                {"class_name": "ripe", "group_id": 2, "type": "polygon", "polygon": [[5, 5], [30, 5], [30, 30]]},
                {"class_name": "stem", "group_id": 2, "type": "polygon", "polygon": [[14, 1], [18, 1], [18, 5], [14, 5]]},
            ]},
        )
        assert save_ann.status_code == 200 and save_ann.get_json()["count"] == 2
        loaded_anns = client.get("/api/annotations/up_test.jpg").get_json()["annotations"]
        assert len(loaded_anns) == 2 and loaded_anns[0]["group_id"] == 2 and loaded_anns[1]["group_id"] == 2

        # Single image export endpoints (.json, .txt, .zip, and invalid args)
        dl_json = client.post("/api/export_image", json={"image_name": "up_test.jpg", "format": "json"})
        assert dl_json.status_code == 200
        dl_lm = json.loads(dl_json.data.decode("utf-8"))
        assert dl_lm["imagePath"] == "up_test.jpg" and len(dl_lm["shapes"]) == 2 and dl_lm["shapes"][0]["group_id"] == 2

        dl_txt = client.post("/api/export_image", json={
            "image_name": "up_test.jpg",
            "format": "txt",
            "classes": ["unripe", "almost_ripe", "ripe", "stem", "calyx"],
        })
        assert dl_txt.status_code == 200
        dl_txt_lines = dl_txt.data.decode("utf-8").strip().splitlines()
        assert len(dl_txt_lines) == 2 and dl_txt_lines[0].startswith("2 ") and dl_txt_lines[1].startswith("3 ")

        dl_both = client.post("/api/export_image", json={
            "image_name": "up_test.jpg",
            "format": "both",
            "classes": "unripe, almost_ripe, ripe, stem, calyx",
        })
        assert dl_both.status_code == 200
        with zipfile.ZipFile(io.BytesIO(dl_both.data), "r") as zf:
            names = set(zf.namelist())
            assert {"up_test.json", "up_test.txt", "classes.txt"}.issubset(names)

        assert client.post("/api/export_image", json={"image_name": "../bad.jpg", "format": "json"}).status_code == 400
        assert client.post("/api/export_image", json={"image_name": "missing.jpg", "format": "json"}).status_code == 404

        # Delete unqualified image (+ .json and .txt cleanup)
        assert client.post("/api/delete_image", json={"image_name": "../bad.jpg"}).status_code == 400
        assert client.post("/api/delete_image", json={"image_name": "bad.txt"}).status_code == 400
        assert client.post("/api/delete_image", json={"image_name": "missing.jpg"}).status_code == 404

        del_up = client.post("/api/upload", data={"file": (_img_bytes(), "del_me.jpg")})
        assert del_up.status_code == 200
        del_save = client.post(
            "/api/annotations/del_me.jpg",
            json={"annotations": [{"class_name": "ripe", "type": "polygon", "polygon": [[4, 4], [20, 4], [20, 20]]}]},
        )
        assert del_save.status_code == 200
        del_ctx = robolabel_app.pm.get_active_context()
        del_img_path = os.path.join(del_ctx["images_dir"], "del_me.jpg")
        del_json_path = os.path.join(del_ctx["annotations_dir"], "del_me.jpg.json")
        del_txt_path = os.path.join(del_ctx["labels_dir"], "del_me.txt")
        assert os.path.exists(del_img_path) and os.path.exists(del_json_path) and os.path.exists(del_txt_path)

        del_resp = client.post("/api/delete_image", json={"image_name": "del_me.jpg"})
        del_json = del_resp.get_json()
        assert del_resp.status_code == 200 and del_json["success"] is True and del_json["deleted_image"] == "del_me.jpg"
        assert not os.path.exists(del_img_path)
        assert not os.path.exists(del_json_path)
        assert not os.path.exists(del_txt_path)
        assert all(it["filename"] != "del_me.jpg" for it in client.get("/api/images").get_json())

        rest_resp = client.post("/api/restore_image", json={})
        rest_json = rest_resp.get_json()
        assert rest_resp.status_code == 200 and rest_json["success"] is True and rest_json["restored_image"] == "del_me.jpg"
        assert os.path.exists(del_img_path) and os.path.exists(del_json_path) and os.path.exists(del_txt_path)
        restored_anns = client.get("/api/annotations/del_me.jpg").get_json()["annotations"]
        assert len(restored_anns) == 1 and restored_anns[0]["class_name"] == "ripe"

        assert client.post("/api/delete_image", json={"image_name": "del_me.jpg"}).status_code == 200
        assert client.post("/api/restore_image", json={}).status_code == 200
        assert client.post("/api/restore_image", json={}).status_code == 400
        assert client.post("/api/delete_image", json={"image_name": "del_me.jpg"}).status_code == 200
        robolabel_app.pm.record_last_image("up_test.jpg")
        rest_bg = client.post("/api/restore_image", json={"update_last_image": False})
        assert rest_bg.status_code == 200 and rest_bg.get_json()["restored_image"] == "del_me.jpg"
        assert robolabel_app.pm.get_active_context()["last_image"] == "up_test.jpg"
        assert client.post("/api/delete_image", json={"image_name": "del_me.jpg"}).status_code == 200

        # Device & polygon editing
        bad_dev = client.post("/api/set_device", json={"device": "invalid"})
        assert bad_dev.status_code == 400 and bad_dev.get_json()["success"] is False
        no_ct_dev = client.post("/api/set_device", data='{"device": "invalid"}')
        assert no_ct_dev.status_code == 400 and no_ct_dev.get_json()["success"] is False

        assert client.post("/api/edit_polygon", json={"polygon": [[0, 0]], "operation": "erase"}).status_code == 400
        erased = client.post("/api/edit_polygon", json={
            "polygon": [[10, 10], [90, 10], [90, 70], [10, 70]],
            "operation": "erase",
            "strokes": [{"x": 50, "y": 40, "radius": 5}],
            "image_width": 100,
            "image_height": 80,
        })
        assert erased.status_code == 200 and len(erased.get_json()["polygons"]) >= 1

        added = client.post("/api/edit_polygon", json={
            "polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
            "operation": "add",
            "add_polygon": [[45, 10], [80, 10], [80, 50], [45, 50]],
            "image_width": 100,
            "image_height": 80,
        })
        assert added.status_code == 200 and len(added.get_json()["polygons"]) == 1

        wiped = client.post("/api/edit_polygon", json={
            "polygon": [[10, 10], [20, 10], [20, 20], [10, 20]],
            "operation": "erase",
            "strokes": [{"x": 15, "y": 15, "radius": 20}],
            "image_width": 40,
            "image_height": 40,
        })
        assert wiped.status_code == 200 and wiped.get_json()["deleted"] is True

        # Segment validation (+ live SAM+CLIP inference if local weight exists)
        assert client.post("/api/segment", json={}).status_code == 400
        assert client.post("/api/segment", json={"image_name": ".", "bbox": [1, 1, 10, 10]}).status_code == 400
        assert client.post("/api/segment", json={"image_name": "../bad.jpg", "bbox": [1, 1, 10, 10]}).status_code == 400
        assert client.post("/api/segment", json={"image_name": "missing.jpg", "bbox": [1, 1, 10, 10]}).status_code == 404
        assert client.post("/api/segment", json={"image_name": "up_test.jpg"}).status_code == 400

        # Project management routes + upload session traversal protection
        assert client.post("/api/projects/analyze_path", json={"source_dir": os.path.join(tmp, "nope")}).status_code == 400
        assert client.post("/api/projects/open_folder", json={"project_dir": src_dir}).status_code == 400
        assert client.post("/api/projects/remove", json={"project_id": "__default__"}).status_code == 400

        good_scan = client.post("/api/projects/analyze_path", json={"source_dir": src_dir})
        assert good_scan.status_code == 200 and good_scan.get_json()["total_images"] == 1

        created = client.post("/api/projects/create_from_path", json={
            "source_dir": src_dir,
            "save_dir": save_dir,
            "project_name": "API Check",
        })
        cdata = created.get_json()
        assert created.status_code == 200 and cdata["success"] is True

        exported = client.get("/api/export")
        assert exported.status_code == 200 and exported.get_json()["success"] is True

        api_flat_dir = os.path.join(tmp, "api_export_flat")
        exported_post = client.post("/api/export", json={
            "format": "both",
            "layout": "flat",
            "output_dir": api_flat_dir,
            "classes": ["unripe", "almost_ripe", "ripe", "stem", "calyx"],
        })
        exp_data = exported_post.get_json()
        assert exported_post.status_code == 200 and exp_data["success"] is True
        assert os.path.exists(os.path.join(api_flat_dir, "sample.json"))
        assert os.path.exists(os.path.join(api_flat_dir, "sample.txt"))
        assert client.post("/api/export", json={
            "format": "json",
            "layout": "flat",
            "output_dir": cdata["project"]["project_dir"],
        }).status_code == 400

        if dm.resolve_weight_path(robolabel_app.WEIGHTS_DIR, robolabel_app.BASE_DIR, "sam2.1_t.pt"):
            seg = client.post("/api/segment", json={
                "image_name": "sample.jpg",
                "model_id": "sam2_tiny",
                "bbox": [25, 15, 95, 85],
                "auto_classify": True,
            })
            sdata = seg.get_json()
            assert seg.status_code == 200 and sdata["success"] is True and len(sdata["polygons"]) >= 1

            seg_pt = client.post("/api/segment", json={
                "image_name": "sample.jpg",
                "model_id": "sam2_tiny",
                "point": [60, 50],
                "auto_classify": True,
            })
            pdata = seg_pt.get_json()
            assert seg_pt.status_code == 200 and pdata["success"] is True and len(pdata["polygons"]) >= 1
            assert pdata["auto_classified"] is True and pdata["confidence"] is not None

        # Upload session chunk + traversal rejection + finalize
        uinit = client.post("/api/projects/upload_init", json={"save_dir": save_dir, "project_name": "Upload Check"})
        sid = uinit.get_json()["session_id"]
        uchunk = client.post("/api/projects/upload_chunk", data={
            "session_id": sid,
            "relative_paths": ["../../evil.jpg", "images/u1.jpg"],
            "files": [(_img_bytes(), "evil.jpg"), (_img_bytes(), "u1.jpg")],
        })
        assert uchunk.status_code == 200 and uchunk.get_json()["saved"] == 1
        ufin = client.post("/api/projects/upload_finalize", json={"session_id": sid})
        assert ufin.status_code == 200 and ufin.get_json()["success"] is True

        pid = cdata["project"]["id"]
        assert client.post("/api/projects/switch", json={"project_id": "__default__"}).status_code == 200
        assert client.post("/api/projects/switch", json={"project_id": pid}).status_code == 200
        assert client.post("/api/projects/open_folder", json={"project_dir": cdata["project"]["project_dir"]}).status_code == 200
        assert client.post("/api/projects/remove", json={"project_id": pid}).status_code == 200
    finally:
        robolabel_app.pm = orig_pm


def check_overlap_detection_and_autofix(tmp):
    # 1. Touching boundary & <= 2.0 sq px overlap must NOT trigger overlap
    touching = [
        {"class_name": "calyx", "type": "polygon", "polygon": [[0, 0], [50, 0], [50, 50], [0, 50]]},
        {"class_name": "ripe", "type": "polygon", "polygon": [[50, 0], [100, 0], [100, 50], [50, 50]]},
    ]
    assert robolabel_app.detect_polygon_overlaps(touching) == []

    tiny_overlap = [
        {"class_name": "calyx", "type": "polygon", "polygon": [[0, 0], [50, 0], [50, 10], [0, 10]]},
        {"class_name": "ripe", "type": "polygon", "polygon": [[49.9, 0], [100, 0], [100, 10], [49.9, 10]]},
    ]
    assert robolabel_app.detect_polygon_overlaps(tiny_overlap) == []

    # 2. Priority 1: calyx drawn BEFORE ripe still takes precedence over ripe
    p1_anns = [
        {"class_name": "calyx", "group_id": 1, "type": "polygon", "polygon": [[10, 10], [50, 10], [50, 40], [10, 40]]},
        {"class_name": "ripe", "group_id": 1, "type": "polygon", "polygon": [[20, 20], [80, 20], [80, 80], [20, 80]]},
    ]
    ov1 = robolabel_app.detect_polygon_overlaps(p1_anns)
    assert len(ov1) == 1 and ov1[0]["pair_label"] == "calyx đè lên ripe"
    fixed1, fc1, sc1 = robolabel_app.autofix_polygon_overlaps(p1_anns)
    assert fc1 == 1 and sc1 == 0 and len(fixed1) == 2
    assert ShapelyPolygon(fixed1[0]["polygon"]).area == ShapelyPolygon(p1_anns[0]["polygon"]).area
    assert round(ShapelyPolygon(fixed1[1]["polygon"]).area, 1) == round(3600.0 - 600.0, 1)
    assert robolabel_app.detect_polygon_overlaps(fixed1) == []

    # 3. Priority 2: stem drawn BEFORE almost_ripe keeps stem 100% and cuts almost_ripe
    p2_anns = [
        {"class_name": "stem", "type": "polygon", "polygon": [[30, 0], [40, 0], [40, 30], [30, 30]]},
        {"class_name": "almost_ripe", "type": "polygon", "polygon": [[20, 20], [60, 20], [60, 70], [20, 70]]},
    ]
    ov2 = robolabel_app.detect_polygon_overlaps(p2_anns)
    assert len(ov2) == 1 and ov2[0]["pair_label"] == "stem đè lên almost_ripe"
    fixed2, fc2, _ = robolabel_app.autofix_polygon_overlaps(p2_anns)
    assert fc2 == 1
    assert ShapelyPolygon(fixed2[0]["polygon"]).area == ShapelyPolygon(p2_anns[0]["polygon"]).area
    assert round(ShapelyPolygon(fixed2[1]["polygon"]).area, 1) == round(2000.0 - 100.0, 1)
    assert robolabel_app.detect_polygon_overlaps(fixed2) == []

    # 4. Same priority: later-drawn polygon (index 1) stays intact, earlier-drawn (index 0) is cut
    same_prio = [
        {"class_name": "ripe", "type": "polygon", "polygon": [[0, 0], [40, 0], [40, 40], [0, 40]]},
        {"class_name": "ripe", "type": "polygon", "polygon": [[20, 0], [60, 0], [60, 40], [20, 40]]},
    ]
    fixed_same, fc_same, _ = robolabel_app.autofix_polygon_overlaps(same_prio)
    assert fc_same == 1 and len(fixed_same) == 2
    assert round(ShapelyPolygon(fixed_same[0]["polygon"]).area, 1) == 800.0
    assert round(ShapelyPolygon(fixed_same[1]["polygon"]).area, 1) == 1600.0
    assert robolabel_app.detect_polygon_overlaps(fixed_same) == []

    # 5. MultiPolygon split: calyx cuts stem into 2 independent polygons preserving group_id and label
    split_anns = [
        {
            "class_name": "stem",
            "label": "stem",
            "group_id": 7,
            "type": "polygon",
            "polygon": [[40, 10], [50, 10], [50, 90], [40, 90]],
        },
        {
            "class_name": "calyx",
            "label": "calyx",
            "group_id": 7,
            "type": "polygon",
            "polygon": [[20, 40], [70, 40], [70, 60], [20, 60]],
        },
    ]
    fixed_split, fc_split, sc_split = robolabel_app.autofix_polygon_overlaps(split_anns)
    assert fc_split == 1 and sc_split == 1 and len(fixed_split) == 3
    stem_pieces = [a for a in fixed_split if a["class_name"] == "stem"]
    assert len(stem_pieces) == 2
    assert all(p.get("group_id") == 7 and p.get("label") == "stem" for p in stem_pieces)
    assert robolabel_app.detect_polygon_overlaps(fixed_split) == []

    # 5b. Completely occluded / sliver polygon must be dropped instead of producing empty/tiny pieces
    occluded_anns = [
        {"class_name": "ripe", "type": "polygon", "polygon": [[20, 20], [50, 20], [50, 50], [20, 50]]},
        {"class_name": "calyx", "type": "polygon", "polygon": [[10, 10], [60, 10], [60, 60], [10, 60]]},
    ]
    fixed_occ, fc_occ, sc_occ = robolabel_app.autofix_polygon_overlaps(occluded_anns)
    assert fc_occ == 1 and sc_occ == 0 and len(fixed_occ) == 1
    assert fixed_occ[0]["class_name"] == "calyx"
    assert robolabel_app.detect_polygon_overlaps(fixed_occ) == []

    # 5c. Non-dict resilience & long-diagonal-edge rounding zero-overlap guarantee
    diag_anns = [
        "invalid_entry",
        {"class_name": "ripe", "type": "polygon", "polygon": [[0, 0], [3000, 0], [3000, 3000], [0, 3000]]},
        {"class_name": "calyx", "type": "polygon", "polygon": [[0.111, 0.333], [2999.777, 1732.444], [1500.555, 2999.888]]},
    ]
    fixed_diag, fc_diag, _ = robolabel_app.autofix_polygon_overlaps(diag_anns)
    assert fc_diag == 1 and len(fixed_diag) >= 2
    assert robolabel_app.detect_polygon_overlaps(fixed_diag) == []

    # 6. Flask API integration: /api/annotations, /api/images, and /api/autofix_overlaps
    base_dir = os.path.join(tmp, "overlap_api_base")
    orig_pm = robolabel_app.pm
    robolabel_app.pm = pm_mod.ProjectManager(base_dir)
    client = robolabel_app.app.test_client()
    try:
        up = client.post("/api/upload", data={"file": (_img_bytes(), "overlap.jpg")})
        assert up.status_code == 200

        save_resp = client.post("/api/annotations/overlap.jpg", json={"annotations": split_anns})
        sjson = save_resp.get_json()
        assert save_resp.status_code == 200
        assert sjson["has_overlap"] is True and sjson["overlap_count"] == 1
        assert sjson["overlaps"][0]["pair_label"] == "calyx đè lên stem"

        imgs = client.get("/api/images").get_json()
        ov_img = next(i for i in imgs if i["filename"] == "overlap.jpg")
        assert ov_img["has_overlap"] is True and ov_img["overlap_count"] == 1

        fix_resp = client.post("/api/autofix_overlaps", json={"image_name": "overlap.jpg"})
        fjson = fix_resp.get_json()
        assert fix_resp.status_code == 200 and fjson["success"] is True
        assert fjson["has_overlap"] is False and fjson["overlap_count"] == 0
        assert fjson["fixed_count"] == 1 and fjson["split_count"] == 1
        assert len(fjson["annotations"]) == 3

        reloaded = client.get("/api/annotations/overlap.jpg").get_json()
        assert reloaded["has_overlap"] is False and len(reloaded["annotations"]) == 3

        imgs_after = client.get("/api/images").get_json()
        ov_img_after = next(i for i in imgs_after if i["filename"] == "overlap.jpg")
        assert ov_img_after["has_overlap"] is False and ov_img_after["overlap_count"] == 0
    finally:
        robolabel_app.pm = orig_pm


def check_freehand_polygon_ui():
    html_path = os.path.join(ROOT_DIR, "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    for marker in ("toolFreehandPoly", "setTool('poly_freehand')", "buildClosedFreehandPolygon", "isDrawingFreehand"):
        assert marker in html, f"missing freehand marker: {marker}"

    scripts = re.findall(r"<script[^>]*>([\s\S]*?)</script>", html)
    main_script = max(scripts, key=len)
    node_exe = shutil.which("node")
    if node_exe:
        js_check = r"""
const fs = require("fs");
const vm = require("vm");
const html = fs.readFileSync(process.argv[1], "utf8");
const scripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const scriptSrc = scripts.sort((a, b) => b.length - a.length)[0];
new vm.Script(scriptSrc);
const helperSrc = scriptSrc.match(/function clampCanvasCoords[\s\S]*?function cancelFreehandPolygon[\s\S]*?\n    \}/)[0];
const ctx = { canvas: { width: 200, height: 150 }, isDrawingFreehand: false, isDragging: false, freehandPoints: [], redraw: () => {}, showToast: () => {} };
vm.runInNewContext(helperSrc, ctx);
const assert = (cond, msg) => { if (!cond) throw new Error(msg); };
assert(ctx.buildClosedFreehandPolygon([], 0.85).length === 0, "empty should return []");
assert(ctx.buildClosedFreehandPolygon([{x:0,y:0},{x:50,y:0},{x:100,y:0}], 0.85).length === 0, "collinear line must be rejected");
assert(ctx.buildClosedFreehandPolygon([{x:10,y:10},{x:11,y:10},{x:11,y:11}], 0.85).length === 0, "tiny area < 4 must be rejected");
const loop = ctx.buildClosedFreehandPolygon([{x:10,y:10},{x:30,y:10},{x:50,y:10},{x:50,y:40},{x:10,y:40},{x:10,y:10}], 0.85);
assert(loop.length === 4, "expected 4 simplified corners, got " + loop.length);
assert(loop[0][0] !== loop[loop.length - 1][0] || loop[0][1] !== loop[loop.length - 1][1], "closing duplicate vertex must be stripped");
const stem = ctx.buildClosedFreehandPolygon([{x:40,y:10},{x:45,y:10},{x:45,y:50},{x:40,y:50}], 0.85);
assert(stem.length === 4, "thin 5x40 stem must be kept");
const clamped = ctx.clampCanvasCoords({x: -12, y: 999});
assert(clamped.x === 0 && clamped.y === 150, "clampCanvasCoords must clamp to canvas size");
"""
        proc = subprocess.run([node_exe, "-e", js_check, html_path], capture_output=True, text=True)
        assert proc.returncode == 0, f"JS freehand check failed: {proc.stderr or proc.stdout}"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        check_device_manager(tmp)
        check_project_manager(tmp)
        check_flask_routes(tmp)
        check_overlap_detection_and_autofix(tmp)
        check_freehand_polygon_ui()
    print("[PASS] All RoboLabel AI self-checks passed.")


if __name__ == "__main__":
    main()
