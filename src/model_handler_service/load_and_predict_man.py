import os
import cv2
import time
from datetime import datetime

from model_handler_service.core.loaders import load_model, predict_class, yolo_predict_crop
from model_handler_service.core.processing import preprocess_image_for_bodytype
from model_handler_service.core.validations import validate_human_image
from model_handler_service.core.config import config
from model_handler_service.core.logger import model_logger
from model_handler_service.color import get_color_tone, detect_clothing



MODEL_CONFIGS = {
    "astin":        ("astin/astinman.h5", ["longsleeve", "shortsleeve", "sleeveless"], "resnet101", 3),
    "pattern":      ("pattern/petternman.h5", ["amudi", "dorosht", "ofoghi", "riz", "sade"], "resnet101", 5),
    "paintane":     ("paintane/mard.h5", ["mbalatane", "mpayintane"], "mobilenet", 2),
    "rise":         ("rise/riseeeeef.h5", ["highrise", "lowrise"], "resnet152_600", 2),
    "shalvar":      ("shalvar/menpants.h5", ["mbaggy", "mcargo", "mcargoshorts", "mmom", "mshorts", "mslimfit", "mstraight"], "resnet101", 7),
    "mnist":        ("under_over/under_over_mobilenet_final.h5", ["Under", "Over"], "mobilenet-v2", 2),
    "tarh_shalvar": ("tarh_shalvar/mmpantsprint.h5", ["mpamudi", "mpdorosht", "mpofoghi", "mpriz", "mpsade"], "resnet101", 5),
    "skirt_pants":  ("skirt_pants/skirt_pants.h5", ["pants", "skirt"], "resnet101", 2),
    "yaghe":        ("yaghe/neckline_classifier_mobilenet.h5", ["classic", "hoodie", "round", "turtleneck", "V_neck"], "mobilenet-v2-softmax", 5),
    "yolo":         ("yolo/best.pt", None, "yolo", 2),
    "body_type":    ("body_type/model_body_type.pth", ["0", "1", "2"], "bodytype", 3),
}

MODELS = {}

def load_all_models():
    """Load all clothing models defined in MODEL_CONFIGS"""
    model_logger.info("Starting to load all men's clothing models")
    start_time = time.time()
    for name, (path, _, base_model, class_num) in MODEL_CONFIGS.items():
        try:
            full_path = str(config.get_model_file_path(path))
            model_logger.debug(f"Loading {name} model: {full_path}")
            MODELS[name] = load_model(full_path, class_num=class_num, base_model=base_model)
        except Exception as e:
            model_logger.error(f"Failed to load model '{name}' from {path}: {e}")
            raise
    total_time = time.time() - start_time
    model_logger.info(f"Successfully loaded all models in {total_time:.2f} seconds")


load_all_models()


# =============================
# process clothing
# =============================

def process_clothing_image(img_path: str) -> dict:
    """
    Process a clothing image and predict various attributes.
    """
    response = {"ok": True, "data": {}, "error": None}
    img = cv2.imread(img_path)

    if img is None:
        return _error_response("IMAGE_LOAD_ERROR", "Failed to load image file")

    # --- Step 1: Clothing detection ---
    try:
        detections = detect_clothing(img_path)
        if not detections:
            model_logger.error("No clothing detected by YOLO validations model.")
            return _error_response("NO_CLOTHING_DETECTED", "لباسی شناسایی نشد")
    except Exception as e:
        model_logger.error(f"YOLO validation detection failed: {e}")
        # ادامه می‌دهیم ولی خطا لاگ میشه

    results = {}

    # --- Step 2: General predictions ---
    results["color_tone"] = _safe_predict(get_color_tone, img_path, "color_tone")
    results["paintane"]   = _safe_predict(predict_class, img, "paintane", MODELS["paintane"], MODEL_CONFIGS["paintane"][1], 224, "paintane")

    # --- Step 3: Conditional predictions ---
    if results["paintane"] == "mbalatane":
        _predict_upper_body(img, img_path, results)
    elif results["paintane"] == "mpayintane":
        _predict_lower_body(img, results)

    # --- Step 4: Final validation ---
    clothing_keys = ["color_tone", "paintane", "astin", "pattern", "yaghe", "rise", "shalvar", "tarh_shalvar", "skirt_pants"]
    if all(results.get(k) is None for k in clothing_keys):
        return _error_response("NO_CLOTHING_DETECTED", "لباسی شناسایی نشد")

    response["data"] = results
    return response


def _predict_upper_body(img, img_path, results):
    """Predictions for upper body clothing (mbalatane)."""
    model_logger.info("Upper body detected → running astin, pattern, yaghe, mnist...")
    try:
        crop_astin, crop_yaghe = yolo_predict_crop(MODELS["yolo"], img_path, img)
    except Exception as e:
        model_logger.error(f"YOLO crop failed: {e}")
        crop_astin, crop_yaghe = None, None

    results["mnist_prediction"] = _safe_predict(predict_class, img, "mnist", MODELS["mnist"], MODEL_CONFIGS["mnist"][1], 224, "mnist")

    if crop_astin is not None:
        results["astin"] = _safe_predict(predict_class, crop_astin, "astin", MODELS["astin"], MODEL_CONFIGS["astin"][1], 300, "astin")
    else:
        results["astin"] = None

    results["pattern"] = _safe_predict(predict_class, img, "pattern", MODELS["pattern"], MODEL_CONFIGS["pattern"][1], 300, "pattern")

    if crop_yaghe is not None:
        results["yaghe"] = _safe_predict(predict_class, crop_yaghe, "yaghe", MODELS["yaghe"], MODEL_CONFIGS["yaghe"][1], 300, "yaghe")
    else:
        results["yaghe"] = None


def _predict_lower_body(img, results):
    """Predictions for lower body clothing (mpayintane)."""
    model_logger.info("Lower body detected → running rise, shalvar, tarh shalvar, skirt/pants...")
    results["rise"]          = _safe_predict(predict_class, img, "rise", MODELS["rise"], MODEL_CONFIGS["rise"][1], 300, "rise")
    results["shalvar"]       = _safe_predict(predict_class, img, "shalvar", MODELS["shalvar"], MODEL_CONFIGS["shalvar"][1], 300, "shalvar")
    results["tarh_shalvar"]  = _safe_predict(predict_class, img, "tarh_shalvar", MODELS["tarh_shalvar"], MODEL_CONFIGS["tarh_shalvar"][1], 300, "tarh_shalvar")
    results["skirt_pants"]   = _safe_predict(predict_class, img, "skirt_pants", MODELS["skirt_pants"], MODEL_CONFIGS["skirt_pants"][1], 300, "skirt_pants")


# =============================
#  Body Type Prediction
# =============================

def get_man_body_type(img_path: str) -> dict:
    """Predict male body type after validation."""
    response = {"ok": True, "data": {}, "error": None}

    try:
        image = preprocess_image_for_bodytype(img_path)
    except Exception as e:
        return _error_response("IMAGE_PREPROCESS_ERROR", str(e))

    errors = validate_human_image(img_path)
    if errors:
        return _error_response("HUMAN_VALIDATION_ERROR", "Failed human body validation", {"human_validation_errors": errors})

    body_type = predict_class(image, MODELS["body_type"], MODEL_CONFIGS["body_type"][1], 300, "bodytype")
    response["data"]["body_type"] = body_type
    return response


# =============================
#  Utilities
# =============================

def _safe_predict(func, *args, name=None, **kwargs):
    """Wrapper for safe prediction with logging."""
    try:
        result = func(*args, **kwargs)
        model_logger.info(f"{name} prediction result: {result}")
        return result
    except Exception as e:
        model_logger.error(f"{name} prediction failed: {e}")
        return None


def _error_response(code, message, extra=None):
    """Unified error response builder."""
    err = {"code": code, "message": message}
    if extra:
        err.update(extra)
    return {"ok": False, "data": None, "error": err}


# =============================
#  Testing Helpers
# =============================

def test_model(model_name: str, sample_path: str):
    """Generic test function for a given model."""
    if model_name not in MODELS:
        print(f"Model '{model_name}' not loaded.")
        return
    img = cv2.imread(sample_path)
    if img is None:
        print(f"Failed to load image: {sample_path}")
        return
    class_names = MODEL_CONFIGS[model_name][1]
    result = predict_class(img, MODELS[model_name], class_names, 300, model_name)
    print(f"{model_name} Prediction: {result}")
