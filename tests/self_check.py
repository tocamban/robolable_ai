"""Self-check covering device_manager, project_manager, Flask routes, and optional live SAM+CLIP smoke."""

import io
import os
import sys
import tempfile

from PIL import Image, ImageDraw

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

    synced = mgr.sync_image_annotations(
        "sample.jpg",
        [{"class_name": "ripe", "type": "polygon", "polygon": [[10, 10], [50, 10], [50, 50], [10, 50]]}],
    )
    assert synced["success"] is True and synced["count"] == 1

    exported = mgr.export_active_yolo()
    assert exported["success"] is True
    assert os.path.exists(exported["data_yaml"])

    mgr.remove_from_registry(pid)
    reopened = mgr.open_existing_project_folder(pdir)
    assert reopened["active_project"]["id"] == pid


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
        assert client.get("/").status_code == 200
        assert client.get("/api/system_info").status_code == 200
        assert len(client.get("/api/models").get_json()) >= 2
        assert "classes" in client.get("/api/classes").get_json()
        assert client.get("/api/images").status_code == 200
        assert client.get("/api/projects").status_code == 200

        # Upload, image fetch, annotation save/load & path traversal check
        assert client.post("/api/upload", data={}).status_code == 400
        up = client.post("/api/upload", data={"file": (_img_bytes(), "up_test.jpg")})
        assert up.status_code == 200 and up.get_json()["filename"] == "up_test.jpg"
        img_resp = client.get("/api/image/up_test.jpg")
        assert img_resp.status_code == 200 and "no-store" in img_resp.headers.get("Cache-Control", "")
        assert client.get("/api/annotations/sub/bad.jpg").status_code == 400

        save_ann = client.post(
            "/api/annotations/up_test.jpg",
            json={"annotations": [{"class_name": "ripe", "type": "polygon", "polygon": [[5, 5], [30, 5], [30, 30]]}]},
        )
        assert save_ann.status_code == 200 and save_ann.get_json()["count"] == 1
        assert len(client.get("/api/annotations/up_test.jpg").get_json()["annotations"]) == 1

        # Device & polygon editing
        bad_dev = client.post("/api/set_device", json={"device": "invalid"})
        assert bad_dev.status_code == 400 and bad_dev.get_json()["success"] is False

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

        if dm.resolve_weight_path(robolabel_app.WEIGHTS_DIR, robolabel_app.BASE_DIR, "sam2.1_t.pt"):
            seg = client.post("/api/segment", json={
                "image_name": "sample.jpg",
                "model_id": "sam2_tiny",
                "bbox": [25, 15, 95, 85],
                "auto_classify": True,
            })
            sdata = seg.get_json()
            assert seg.status_code == 200 and sdata["success"] is True and len(sdata["polygons"]) >= 1

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


def main():
    with tempfile.TemporaryDirectory() as tmp:
        check_device_manager(tmp)
        check_project_manager(tmp)
        check_flask_routes(tmp)
    print("[PASS] All RoboLabel AI self-checks passed.")


if __name__ == "__main__":
    main()
