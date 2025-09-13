
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import time
from model_handler_service.core.loaders import load_model, predict_class, yolo_predict_crop
from model_handler_service.core.processing import preprocess_image_for_bodytype
from model_handler_service.core.validations import validate_human_image
from model_handler_service.core.logger import model_logger
from model_handler_service.color import detect_clothing, get_color_tone
from model_handler_service.core.config import config
import torch

# Define model paths using the MODEL_PATH from config
# This will use the path defined in the .env file

# Women's clothing models
model_astin_path = str(config.get_model_file_path('astin/astinwoman.h5'))
model_patern_path = str(config.get_model_file_path('pattern/petternwoman.h5'))
model_paintane_path = str(config.get_model_file_path('paintane/p_t_b_woman_1Mar_10_45.h5'))
model_rise_path = str(config.get_model_file_path('rise/riseeeeef.h5'))
model_shalvar_path = str(config.get_model_file_path('shalvar/womenpants.h5'))
model_tarh_shalvar_path = str(config.get_model_file_path('tarh_shalvar/wwpantsprint.h5'))
model_skirt_pants_path = str(config.get_model_file_path('skirt_pants/skirt_pants.h5'))
model_yaghe_path = str(config.get_model_file_path('yaghe/yaghewoman101A.h5'))
model_skirt_print_path = str(config.get_model_file_path('skirt_print/skirt_print.h5'))
model_skirt_type_path = str(config.get_model_file_path('skirt_type/skirttt_types.h5'))
model_mnist_path = str(config.get_model_file_path('under_over/under_over_mobilenet_final.h5'))
model_body_type_path = str(config.get_model_file_path('body_type/model_women.pt'))
model_yolo_path = str(config.get_model_file_path('yolo/best.pt'))

# Six model paths
model_balted_path = str(config.get_model_file_path('6model/belted.h5'))
model_cowl_path = str(config.get_model_file_path('6model/cowl.h5'))
model_empire_path = str(config.get_model_file_path('6model/empire.h5'))
model_loose_path = str(config.get_model_file_path('6model/loose.h5'))
model_peplum_path = str(config.get_model_file_path('6model/peplum.h5'))
model_wrap_path = str(config.get_model_file_path('6model/wrap.h5'))


# Load models globally
model_logger.info("Starting to load all women's clothing models")
start_time = time.time()

try:
    model_logger.debug(f"Loading model: {model_astin_path}")
    model_astin = load_model(model_astin_path, class_num=6, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_patern_path}")
    model_patern = load_model(model_patern_path, class_num=5, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_paintane_path}")
    model_paintane = load_model(model_paintane_path, class_num=3, base_model="mobilenet-v2-pt")
    model_logger.debug(f"Loading model: {model_rise_path}")
    model_rise = load_model(model_rise_path, class_num=2, base_model="resnet152_600")
    model_logger.debug(f"Loading model: {model_shalvar_path}")
    model_shalvar = load_model(model_shalvar_path, class_num=8, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_tarh_shalvar_path}")
    model_tarh_shalvar = load_model(model_tarh_shalvar_path, class_num=5, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_skirt_pants_path}")
    model_skirt_pants = load_model(model_skirt_pants_path, class_num=2, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_yaghe_path}")
    model_yaghe = load_model(model_yaghe_path, class_num=11, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_skirt_print_path}")
    model_skirt_print = load_model(model_skirt_print_path, class_num=5, base_model="resnet101_30_unit")
    model_logger.debug(f"Loading model: {model_skirt_type_path}")
    model_skirt_type = load_model(model_skirt_type_path, class_num=7, base_model="resnet101_30_unit")
    model_logger.debug(f"Loading model: {model_mnist_path}")
    model_mnist = load_model(model_mnist_path, class_num=2, base_model="mobilenet-v2")

    model_logger.debug(f"Loading PyTorch body type model: {model_body_type_path}")
    model_body_type = torch.load(model_body_type_path, map_location=torch.device('cpu'))  # Load PyTorch model body type (arian)

    model_logger.debug(f"Loading model: {model_yolo_path}")
    model_yolo = load_model(model_yolo_path, class_num=2, base_model="yolo")
    model_logger.debug(f"Loading model: {model_balted_path}")
    model_balted = load_model(model_balted_path, class_num=2, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_cowl_path}")
    model_cowl = load_model(model_cowl_path, class_num=2, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_empire_path}")
    model_empire = load_model(model_empire_path, class_num=2, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_loose_path}")
    model_loose = load_model(model_loose_path, class_num=2, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_peplum_path}")
    model_peplum = load_model(model_peplum_path, class_num=2, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_wrap_path}")
    model_wrap = load_model(model_wrap_path, class_num=2, base_model="resnet101")

    total_time = time.time() - start_time
    model_logger.info(f"Successfully loaded all women's clothing models in {total_time:.2f} seconds")
except Exception as e:
    model_logger.error(f"Failed to load women's clothing models: {str(e)}")
    raise


def process_woman_clothing_image(image_path):
    """
    Processes a woman's clothing image to predict various attributes such as color tone,
    clothing type (paintane), sleeve type (astin), pattern, neckline (yaghe),
    rise, shalvar type, shalvar pattern, and skirt/pants classification.

    Args:
        image_path (str): The path to the clothing image file.

    Returns:
        dict: A dictionary containing the prediction results for various clothing attributes including:
            - color_tone: The dominant color tone of the clothing
            - mnist_prediction: Classification as "Under" or "Over" clothing
            - paintane: Clothing type (fbalatane, fpaintane, ftamamtane)
            - astin: Sleeve type (if applicable)
            - pattern: Pattern type of the clothing
            - yaghe: Neckline type (if applicable)
            - rise: Rise type for pants (if applicable)
            - shalvar: Pants/shalvar type (if applicable)
            - tarh_shalvar: Pants pattern (if applicable)
            - skirt_pants: Classification as skirt or pants (if applicable)
            - skirt_print: Skirt pattern (if applicable)
            - skirt_type: Skirt type (if applicable)
    """
    image = cv2.imread(image_path)
    
    response = {
        "ok": True,
        "data": {},
        "error": None
    }

    # Validate image was loaded successfully
    if image is None:
        response["ok"] = False
        response["error"] = {
            "code": "IMAGE_LOAD_ERROR",
            "message": "Failed to load image file",
        }
        return response

    # Validate clothing is visible in image
    clothing_validation_errors = []
    # TODO: Add actual clothing validation logic here
    if len(clothing_validation_errors) > 0:
        response["ok"] = False
        response["error"] = {
            "code": "CLOTHING_VALIDATION_ERROR",
            "message": "Failed clothing validation checks",
            "validation_errors": clothing_validation_errors
        }
        return response
    # Preliminary clothing detection using validations/clothes/detect_clothes.pt
    try:
        detections = detect_clothing(image_path)
        if not detections:
            model_logger.error("No clothing detected by YOLO validations model. Returning error response.")
            response["ok"] = False
            response["data"] = None
            response["error"] = {
                "code": "NO_CLOTHING_DETECTED",
                "message": "No clothing detected in the image (YOLO validations model found nothing)"
            }
            return response
    except Exception as e:
        model_logger.error(f"Validations YOLO detection failed: {e}")


    from model_handler_service.core.logger import model_logger
    model_logger.info(f"Processing image: {image_path}")
    results = {}

    # 2. General Predictions (Independent of clothing type)
    model_logger.info("Start color_tone prediction")
    try:
        results["color_tone"] = get_color_tone(image_path)
        model_logger.info(f"Color tone prediction result: {results.get('color_tone')}")
    except Exception as e:
        model_logger.error(f"Color tone prediction failed: {e}")
        results["color_tone"] = None

    model_logger.info("Start paintane prediction")
    try:
        results["paintane"] = predict_class(image, model_paintane, ["fbalatane", "fpaintane", "ftamamtane"], 224, "paintane")
        model_logger.info(f"Paintane prediction result: {results.get('paintane')}")
    except Exception as e:
        model_logger.error(f"Paintane prediction failed: {e}")
        results["paintane"] = None

    # 3. Conditional Predictions based on 'paintane' (Clothing Type)
    # Upper body یا full body clothing
    if results["paintane"] in ["fbalatane", "ftamamtane"]:
        model_logger.info("Detected 'fbalatane' or 'ftamamtane'. Proceeding with upper body predictions.")
        try:
            results["mnist_prediction"] = predict_class(image, model_mnist, ["Under", "Over"], 224, "mnist")
            model_logger.info(f"MNIST prediction result: {results.get('mnist_prediction')}")
        except Exception as e:
            model_logger.error(f"MNIST prediction failed: {e}")
            results["mnist_prediction"] = None

        try:
            crop_image_astin, crop_image_yaghe = yolo_predict_crop(model=model_yolo, image_path=image_path)
            model_logger.info(f"YOLO detection completed. astin crop: {crop_image_astin is not None}, yaghe crop: {crop_image_yaghe is not None}")
        except Exception as e:
            model_logger.error(f"YOLO crop failed: {e}")
            crop_image_astin, crop_image_yaghe = None, None

        # شرط جدید: اگر هیچ لباس پیدا نشد (هر دو crop None)
        if crop_image_astin is None and crop_image_yaghe is None:
            model_logger.error("YOLO did not detect any clothing. Returning error response.")
            response["ok"] = False
            response["data"] = None
            response["error"] = {
                "code": "NO_CLOTHING_DETECTED",
                "message": "No clothing detected in the image (YOLO found nothing)"
            }
            return response

        # Astin prediction only if crop is not None
        if crop_image_astin is not None:
            try:
                results["astin"] = predict_class(crop_image_astin, model_astin, ['bottompuffy', "fhalfsleeve", "flongsleeve", "fshortsleeve", "fsleeveless", "toppuffy"], 300, "astin")
                model_logger.info(f"Astin prediction result: {results.get('astin')}")
            except Exception as e:
                model_logger.error(f"Astin prediction failed: {e}")
                results["astin"] = None
        else:
            model_logger.error("Astin crop is None, skipping astin prediction.")
            results["astin"] = None

        try:
            results["pattern"] = predict_class(image, model_patern, ["dorosht", "rahrahamudi", "rahrahofoghi", "riz", "sade"], 300, "pattern")
            model_logger.info(f"Pattern prediction result: {results.get('pattern')}")
        except Exception as e:
            model_logger.error(f"Pattern prediction failed: {e}")
            results["pattern"] = None

        # Yaghe prediction only if crop is not None
        if crop_image_yaghe is not None:
            try:
                results["yaghe"] = predict_class(crop_image_yaghe, model_yaghe, ["boatneck", "classic", "halter", "hoodie", "of_the_shoulder", "one_shoulder", "round", "squer", "sweatheart", 'turtleneck', "v_neck"], 300, "yaghe")
                model_logger.info(f"Yaghe prediction result: {results.get('yaghe')}")
            except Exception as e:
                model_logger.error(f"Yaghe prediction failed: {e}")
                results["yaghe"] = None
        else:
            model_logger.error("Yaghe crop is None, skipping yaghe prediction.")
            results["yaghe"] = None

    # Lower body or full body clothing
    if results["paintane"] in ["fpaintane", "ftamamtane"]:
        model_logger.info("Detected 'fpaintane' or 'ftamamtane'. Proceeding with lower body predictions.")
        try:
            results["rise"] = predict_class(image, model_rise, ["highrise", "lowrise"], 300, "rise")
            model_logger.info(f"Rise prediction result: {results.get('rise')}")
        except Exception as e:
            model_logger.error(f"Rise prediction failed: {e}")
            results["rise"] = None

        try:
            results["shalvar"] = predict_class(image, model_shalvar, ["wbaggy", "wbootcut", "wcargo", "wcargoshorts", "wmom", "wshorts", "wskinny", "wstraight"], 300, "noeshalvar")
            model_logger.info(f"Shalvar prediction result: {results.get('shalvar')}")
        except Exception as e:
            model_logger.error(f"Shalvar prediction failed: {e}")
            results["shalvar"] = None

        try:
            results["tarh_shalvar"] = predict_class(image, model_tarh_shalvar, ["wpamudi", "wpdorosht", "wpofoghi", "wpriz", "wpsade"], 300, "tarhshalvar")
            model_logger.info(f"Tarh shalvar prediction result: {results.get('tarh_shalvar')}")
        except Exception as e:
            model_logger.error(f"Tarh shalvar prediction failed: {e}")
            results["tarh_shalvar"] = None

        try:
            results["skirt_and_pants"] = predict_class(image, model_skirt_pants, ["pants", "skirt"], 300, "skirt and pants")
            model_logger.info(f"Skirt and pants prediction result: {results.get('skirt_and_pants')}")
        except Exception as e:
            model_logger.error(f"Skirt and pants prediction failed: {e}")
            results["skirt_and_pants"] = None

        try:
            results["skirt_print"] = predict_class(image, model_skirt_print, ["skirtamudi", "skirtdorosht", "skirtofoghi", "skirtriz", "skirtsade"], 300, "skirt and pants")
            model_logger.info(f"Skirt print prediction result: {results.get('skirt_print')}")
        except Exception as e:
            model_logger.error(f"Skirt print prediction failed: {e}")
            results["skirt_print"] = None

        try:
            results["skirt_type"] = predict_class(image, model_skirt_type, ["alineskirt", "balloonskirt", "mermaidskirt", "miniskirt", "pencilskirt", "shortaskirt", "wrapskirt"], 300, "model_skirt_type")
            model_logger.info(f"Skirt type prediction result: {results.get('skirt_type')}")
        except Exception as e:
            model_logger.error(f"Skirt type prediction failed: {e}")
            results["skirt_type"] = None

    # Check if no clothing detected (کلیدهای مهم None باشند)
    clothing_keys = ["color_tone", "paintane", "astin", "pattern", "yaghe", "rise", "shalvar", "tarh_shalvar", "skirt_and_pants"]
    clothing_none = all(results.get(k) is None for k in clothing_keys)
    if clothing_none:
        model_logger.error("No clothing detected in the image. Returning error response.")
        response["ok"] = False
        response["data"] = None
        response["error"] = {
            "code": "NO_CLOTHING_DETECTED",
            "message": "No clothing detected in the image"
        }
        return response
    response["data"] = results
    return response


def get_body_type_female(image_path):
    
    response = {
        "ok": True,
        "data": {},
        "error": None
    }

    # Loading and preprocessing image
    try:
        image = preprocess_image_for_bodytype(image_path)
    except Exception as e:
        response["ok"] = False
        response["error"] = {
            "code": "IMAGE_PREPROCESS_ERROR",
            "message": str(e),
        }
        return response

    # Human validation
    human_validation_errors = validate_human_image(image_path)
    if len(human_validation_errors) > 0:
        response["ok"] = False
        response["error"] = {
            "code": "HUMAN_VALIDATION_ERROR", 
            "message": "Failed human body validation",
            "human_validation_errors": human_validation_errors
        }
        return response

    # Only predict if validation passed
    body_type = predict_class(
        image,
        model_body_type,
        ["11", "21", "31", "41", "51"],
        300,
        "bodytype"
    )
    
    response["data"]["body_type"] = body_type
    return response


def process_six_model_predictions(image_path):
    """
    Processes a woman's clothing image to predict various style attributes.
    
    Args:
        image_path (str): The path to the clothing image file.
        
    Returns:
        dict: A dictionary containing the prediction results for various clothing attributes including:
            - balted: Whether the clothing is belted or not
            - cowl: Whether the clothing has a cowl neck or not
            - empire: Whether the clothing has an empire waist or not
            - loose: Whether the clothing is loose or snatched
            - wrap: Whether the clothing is wrap style or not
            - peplum: Whether the clothing has a peplum or not
    """
    from model_handler_service.core.logger import model_logger
    image = cv2.imread(image_path)
    results = {}
    model_logger.info("Start 6-model predictions")
    try:
        results["balted"] = predict_class(image, model_balted, ["balted", "notbalted"], 300, "balted")
        model_logger.info(f"Balted prediction result: {results.get('balted')}")
    except Exception as e:
        model_logger.error(f"Balted prediction failed: {e}")
        results["balted"] = None
    try:
        results["cowl"] = predict_class(image, model_cowl, ["cowl", "notcowl"], 300, "cowl")
        model_logger.info(f"Cowl prediction result: {results.get('cowl')}")
    except Exception as e:
        model_logger.error(f"Cowl prediction failed: {e}")
        results["cowl"] = None
    try:
        results["empire"] = predict_class(image, model_empire, ["empire", "notempire"], 300, "empire")
        model_logger.info(f"Empire prediction result: {results.get('empire')}")
    except Exception as e:
        model_logger.error(f"Empire prediction failed: {e}")
        results["empire"] = None
    try:
        results["loose"] = predict_class(image, model_loose, ["losse", "snatched"], 300, "loose")
        model_logger.info(f"Loose prediction result: {results.get('loose')}")
    except Exception as e:
        model_logger.error(f"Loose prediction failed: {e}")
        results["loose"] = None
    try:
        results["wrap"] = predict_class(image, model_wrap, ["notwrap", "wrap"], 300, "wrap")
        model_logger.info(f"Wrap prediction result: {results.get('wrap')}")
    except Exception as e:
        model_logger.error(f"Wrap prediction failed: {e}")
        results["wrap"] = None
    try:
        results["peplum"] = predict_class(image, model_peplum, ["notpeplum", "peplum"], 300, "peplum")
        model_logger.info(f"Peplum prediction result: {results.get('peplum')}")
    except Exception as e:
        model_logger.error(f"Peplum prediction failed: {e}")
        results["peplum"] = None
    model_logger.info(f"Returning 6 model results: {results}")
    return results
