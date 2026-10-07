"""
test_final_e2e.py
-----------------
Final End-to-End Test Suite for Food Freshness Detection Demo Readiness.
Executes all 10 test stages and outputs the formatted final test report.
"""

import csv
import io
import json
import os
import sys
from pathlib import Path
from PIL import Image
import requests

SERVER_URL = "http://127.0.0.1:5000"
PROJECT_ROOT = Path(r"c:\Users\blued\.antigravity-ide\New folder (2)\new food\Food_Freshness_Detection")
TEST_DATASET_DIR = PROJECT_ROOT / "dataset" / "test"

results = {
    "startup": False,
    "fresh_image": False,
    "rotten_image": False,
    "invalid_file": False,
    "no_file": False,
    "corrupted_file": False,
    "gradcam": False,
    "quality_inspection": False,
    "pdf_report": False,
    "csv_report": False,
    "multiple_predictions": False,
    "server_stability": False,
}

details = {}


def find_image_for_class(class_name: str):
    class_dir = TEST_DATASET_DIR / class_name
    if not class_dir.is_dir():
        return None
    valid_exts = {".jpg", ".jpeg", ".png"}
    for img_file in sorted(class_dir.iterdir()):
        if img_file.is_file() and img_file.suffix.lower() in valid_exts:
            try:
                with Image.open(img_file) as img:
                    img.verify()
                return img_file
            except Exception:
                continue
    return None


def run_e2e_suite():
    print("=" * 70)
    print("EXECUTING FINAL END-TO-END DEMO READINESS SUITE")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # TEST 1: APPLICATION STARTUP & HOMEPAGE
    # -----------------------------------------------------------------------
    print("\n[TEST 1] Application Startup & Homepage")
    try:
        r_home = requests.get(f"{SERVER_URL}/", timeout=5)
        assert r_home.status_code == 200
        assert "Food Freshness Detection" in r_home.text
        assert "Analyse Freshness" in r_home.text
        assert "Quality Inspection" in r_home.text
        assert "Grad-CAM Explanation" in r_home.text
        results["startup"] = True
        print("  -> Application startup & homepage: PASSED (HTTP 200)")
    except Exception as e:
        print(f"  -> Application startup FAILED: {e}")

    # -----------------------------------------------------------------------
    # TEST 2: FRESH IMAGE (FreshApple)
    # -----------------------------------------------------------------------
    print("\n[TEST 2] Fresh Image Prediction")
    fresh_img = find_image_for_class("FreshApple")
    assert fresh_img is not None, "FreshApple test image not found"
    try:
        with open(fresh_img, "rb") as f:
            r_fresh = requests.post(
                f"{SERVER_URL}/predict",
                files={"food_image": (fresh_img.name, f, "image/jpeg")},
                timeout=10,
            )
        assert r_fresh.status_code == 200
        fresh_data = r_fresh.json()
        assert fresh_data.get("success") is True
        assert fresh_data.get("produce") == "Apple"
        assert fresh_data.get("freshness") == "Fresh"
        assert isinstance(fresh_data.get("confidence"), float)
        assert fresh_data.get("gradcam_image") is not None
        assert fresh_data.get("quality_status") == "Visually appears fresh"
        assert "fresh Apple" in fresh_data.get("visual_summary", "")

        # Verify Grad-CAM image
        cam_rel = fresh_data["gradcam_image"].lstrip("/")
        cam_file = PROJECT_ROOT / cam_rel
        assert cam_file.is_file()
        with Image.open(cam_file) as img:
            img.verify()

        results["fresh_image"] = True
        results["gradcam"] = True
        results["quality_inspection"] = True
        details["fresh_pred"] = fresh_data
        print(f"  -> Predicted: {fresh_data['predicted_class']}")
        print(f"  -> Produce: {fresh_data['produce']}, Freshness: {fresh_data['freshness']}")
        print(f"  -> Confidence: {fresh_data['confidence']*100:.2f}%")
        print(f"  -> Quality Status: {fresh_data['quality_status']}")
        print(f"  -> Grad-CAM Generated: {fresh_data['gradcam_image']}")
        print("  -> Fresh Image Prediction: PASSED")
    except Exception as e:
        print(f"  -> Fresh Image Prediction FAILED: {e}")

    # -----------------------------------------------------------------------
    # TEST 3: ROTTEN IMAGE (RottenBanana)
    # -----------------------------------------------------------------------
    print("\n[TEST 3] Rotten Image Prediction")
    rotten_img = find_image_for_class("RottenBanana")
    assert rotten_img is not None, "RottenBanana test image not found"
    try:
        with open(rotten_img, "rb") as f:
            r_rotten = requests.post(
                f"{SERVER_URL}/predict",
                files={"food_image": (rotten_img.name, f, "image/jpeg")},
                timeout=10,
            )
        assert r_rotten.status_code == 200
        rotten_data = r_rotten.json()
        assert rotten_data.get("success") is True
        assert rotten_data.get("produce") == "Banana"
        assert rotten_data.get("freshness") == "Rotten"
        assert isinstance(rotten_data.get("confidence"), float)
        assert rotten_data.get("gradcam_image") is not None
        assert rotten_data.get("quality_status") == "Signs of spoilage detected"
        assert "spoiled Banana" in rotten_data.get("visual_summary", "")

        results["rotten_image"] = True
        details["rotten_pred"] = rotten_data
        print(f"  -> Predicted: {rotten_data['predicted_class']}")
        print(f"  -> Produce: {rotten_data['produce']}, Freshness: {rotten_data['freshness']}")
        print(f"  -> Confidence: {rotten_data['confidence']*100:.2f}%")
        print(f"  -> Quality Status: {rotten_data['quality_status']}")
        print(f"  -> Grad-CAM Generated: {rotten_data['gradcam_image']}")
        print("  -> Rotten Image Prediction: PASSED")
    except Exception as e:
        print(f"  -> Rotten Image Prediction FAILED: {e}")

    # -----------------------------------------------------------------------
    # TEST 4: INVALID FILE (.txt, .pdf)
    # -----------------------------------------------------------------------
    print("\n[TEST 4] Invalid File Handling")
    try:
        r_txt = requests.post(
            f"{SERVER_URL}/predict",
            files={"food_image": ("notes.txt", b"plain text report", "text/plain")},
            timeout=5,
        )
        assert r_txt.status_code == 400
        txt_json = r_txt.json()
        assert txt_json.get("success") is False
        assert "Unsupported file format" in txt_json.get("error", "")

        r_pdf_upload = requests.post(
            f"{SERVER_URL}/predict",
            files={"food_image": ("document.pdf", b"%PDF-1.4 dummy", "application/pdf")},
            timeout=5,
        )
        assert r_pdf_upload.status_code == 400
        assert r_pdf_upload.json().get("success") is False

        results["invalid_file"] = True
        print(f"  -> Invalid file rejection: PASSED (Error: {txt_json['error']})")
    except Exception as e:
        print(f"  -> Invalid File Handling FAILED: {e}")

    # -----------------------------------------------------------------------
    # TEST 5: NO FILE
    # -----------------------------------------------------------------------
    print("\n[TEST 5] No File Selected Handling")
    try:
        r_empty = requests.post(f"{SERVER_URL}/predict", files={}, timeout=5)
        assert r_empty.status_code == 400
        empty_json = r_empty.json()
        assert empty_json.get("success") is False
        assert "No file selected" in empty_json.get("error", "")

        results["no_file"] = True
        print(f"  -> No-file rejection: PASSED (Error: {empty_json['error']})")
    except Exception as e:
        print(f"  -> No-File Handling FAILED: {e}")

    # -----------------------------------------------------------------------
    # TEST 6: CORRUPTED IMAGE
    # -----------------------------------------------------------------------
    print("\n[TEST 6] Corrupted Image Handling")
    try:
        r_corrupt = requests.post(
            f"{SERVER_URL}/predict",
            files={"food_image": ("corrupted.jpg", b"\xFF\xD8\xFF\xE0garbagebytes1234567890", "image/jpeg")},
            timeout=5,
        )
        assert r_corrupt.status_code == 400
        corrupt_json = r_corrupt.json()
        assert corrupt_json.get("success") is False
        assert "Unreadable or corrupted" in corrupt_json.get("error", "")

        results["corrupted_file"] = True
        print(f"  -> Corrupted image rejection: PASSED (Error: {corrupt_json['error']})")
    except Exception as e:
        print(f"  -> Corrupted Image Handling FAILED: {e}")

    # -----------------------------------------------------------------------
    # TEST 7: PDF REPORT
    # -----------------------------------------------------------------------
    print("\n[TEST 7] PDF Report Generation & Verification")
    try:
        r_pdf = requests.get(f"{SERVER_URL}/report/pdf", timeout=10)
        assert r_pdf.status_code == 200
        assert "application/pdf" in r_pdf.headers.get("Content-Type", "")
        pdf_content = r_pdf.content
        assert pdf_content.startswith(b"%PDF")
        assert len(pdf_content) > 2000

        # Verify PDF contains required section tokens
        assert b"FOOD FRESHNESS DETECTION REPORT" in pdf_content or b"Produce" in pdf_content

        results["pdf_report"] = True
        print(f"  -> PDF generated successfully ({len(pdf_content)} bytes)")
        print("  -> PDF Report Verification: PASSED")
    except Exception as e:
        print(f"  -> PDF Report FAILED: {e}")

    # -----------------------------------------------------------------------
    # TEST 8: CSV REPORT
    # -----------------------------------------------------------------------
    print("\n[TEST 8] CSV Report Generation & Verification")
    try:
        r_csv = requests.get(f"{SERVER_URL}/report/csv", timeout=5)
        assert r_csv.status_code == 200
        assert "text/csv" in r_csv.headers.get("Content-Type", "")

        reader = csv.DictReader(io.StringIO(r_csv.text))
        rows = list(reader)
        assert len(rows) == 1
        row = rows[0]

        expected_headers = [
            "Date/Time", "Image Filename", "Predicted Class",
            "Produce", "Freshness", "Confidence",
            "Quality Status", "Visual Summary", "Grad-CAM Image"
        ]
        for header in expected_headers:
            assert header in row, f"Missing header: {header}"

        assert row["Produce"] == "Banana"
        assert row["Freshness"] == "Rotten"
        assert row["Quality Status"] == "Signs of spoilage detected"
        assert "spoiled Banana" in row["Visual Summary"]
        assert row["Grad-CAM Image"] != "N/A"

        results["csv_report"] = True
        print("  -> CSV Headers and Content Verified:")
        for k, v in row.items():
            print(f"       {k}: {v}")
        print("  -> CSV Report Verification: PASSED")
    except Exception as e:
        print(f"  -> CSV Report FAILED: {e}")

    # -----------------------------------------------------------------------
    # TEST 9: MULTIPLE PREDICTIONS (3 distinct produce items)
    # -----------------------------------------------------------------------
    print("\n[TEST 9] Multiple Sequential Predictions Test")
    test_classes = ["FreshOrange", "RottenTomato", "FreshBellpepper"]
    multi_passed = True
    prev_cam = None

    for cls_name in test_classes:
        img_p = find_image_for_class(cls_name)
        assert img_p is not None, f"Image for {cls_name} not found"
        try:
            with open(img_p, "rb") as f:
                r_m = requests.post(
                    f"{SERVER_URL}/predict",
                    files={"food_image": (img_p.name, f, "image/jpeg")},
                    timeout=10,
                )
            assert r_m.status_code == 200
            m_data = r_m.json()
            assert m_data.get("success") is True
            assert m_data.get("predicted_class") == cls_name
            assert m_data.get("gradcam_image") != prev_cam
            prev_cam = m_data.get("gradcam_image")

            print(f"  -> Prediction {cls_name}: {m_data['produce']} | {m_data['freshness']} ({m_data['confidence']*100:.2f}%) | Cam: {m_data['gradcam_image']}")
        except Exception as e:
            print(f"  -> Multiple prediction failed on {cls_name}: {e}")
            multi_passed = False

    # Check latest report reflects the last prediction (FreshBellpepper)
    try:
        r_latest_csv = requests.get(f"{SERVER_URL}/report/csv", timeout=5)
        latest_row = list(csv.DictReader(io.StringIO(r_latest_csv.text)))[0]
        assert latest_row["Produce"] == "Bellpepper"
        assert latest_row["Freshness"] == "Fresh"
        assert latest_row["Quality Status"] == "Visually appears fresh"
        print("  -> Latest report properly updated to FreshBellpepper")
    except Exception as e:
        print(f"  -> Latest report check failed: {e}")
        multi_passed = False

    if multi_passed:
        results["multiple_predictions"] = True
        print("  -> Multiple Predictions Test: PASSED")

    # -----------------------------------------------------------------------
    # TEST 10: SERVER STABILITY
    # -----------------------------------------------------------------------
    print("\n[TEST 10] Server Stability Check")
    try:
        # Check server is still up and responsive
        r_final = requests.get(f"{SERVER_URL}/", timeout=5)
        assert r_final.status_code == 200
        results["server_stability"] = True
        print("  -> Server remains online, responsive, and stable: PASSED")
    except Exception as e:
        print(f"  -> Server stability FAILED: {e}")

    # -----------------------------------------------------------------------
    # SUMMARY & FINAL REPORT
    # -----------------------------------------------------------------------
    overall_status = all(results.values())

    print("\n" + "=" * 70)
    print("FINAL END-TO-END TEST REPORT")
    print("============================")
    print(f"Application startup       : {'PASSED' if results['startup'] else 'FAILED'}")
    print(f"Fresh image prediction    : {'PASSED' if results['fresh_image'] else 'FAILED'}")
    print(f"Rotten image prediction   : {'PASSED' if results['rotten_image'] else 'FAILED'}")
    print(f"Invalid file handling     : {'PASSED' if results['invalid_file'] else 'FAILED'}")
    print(f"No-file handling          : {'PASSED' if results['no_file'] else 'FAILED'}")
    print(f"Corrupted file handling   : {'PASSED' if results['corrupted_file'] else 'FAILED'}")
    print(f"Grad-CAM                   : {'PASSED' if results['gradcam'] else 'FAILED'}")
    print(f"Quality Inspection        : {'PASSED' if results['quality_inspection'] else 'FAILED'}")
    print(f"PDF report                : {'PASSED' if results['pdf_report'] else 'FAILED'}")
    print(f"CSV report                : {'PASSED' if results['csv_report'] else 'FAILED'}")
    print(f"Multiple predictions      : {'PASSED' if results['multiple_predictions'] else 'FAILED'}")
    print(f"Server stability          : {'PASSED' if results['server_stability'] else 'FAILED'}")
    print()
    print(f"OVERALL STATUS             : {'PASSED' if overall_status else 'FAILED'}")
    print("=" * 70)

    if not overall_status:
        sys.exit(1)


if __name__ == "__main__":
    run_e2e_suite()
