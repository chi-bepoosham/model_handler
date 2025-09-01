
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import time
from model_handler_service.core.loaders import load_model, predict_class, yolo_predict_crop
from model_handler_service.core.processing import preprocess_image_for_bodytype
from model_handler_service.core.validations import validate_human_image
from model_handler_service.core.config import config
from model_handler_service.core.logger import model_logger
from model_handler_service.color import get_color_tone, detect_clothing

# Define model paths using the MODEL_PATH from config
# This will use the path defined in the .env file

# Men's clothing models
model_astin_path = str(config.get_model_file_path('astin/astinman.h5'))
model_patern_path = str(config.get_model_file_path('pattern/petternman.h5'))
model_paintane_path = str(config.get_model_file_path('paintane/mard.h5'))
model_rise_path = str(config.get_model_file_path('rise/riseeeeef.h5'))
model_shalvar_path = str(config.get_model_file_path('shalvar/menpants.h5'))
model_mnist_path = str(config.get_model_file_path('under_over/under_over_mobilenet_final.h5'))
model_tarh_shalvar_path = str(config.get_model_file_path('tarh_shalvar/mmpantsprint.h5'))
model_skirt_pants_path = str(config.get_model_file_path('skirt_pants/skirt_pants.h5'))
model_yaghe_path = str(config.get_model_file_path('yaghe/neckline_classifier_mobilenet.h5'))
model_yolo_path = str(config.get_model_file_path('yolo/best.pt'))
model_body_type_path = str(config.get_model_file_path('body_type/model_body_type.pth'))

# Load models globally
model_logger.info("Starting to load all men's clothing models")
start_time = time.time()

try:
    model_logger.debug(f"Loading model: {model_astin_path}")
    model_astin = load_model(model_astin_path, class_num=3, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_patern_path}")
    model_patern = load_model(model_patern_path, class_num=5, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_paintane_path}")
    model_paintane = load_model(model_paintane_path, class_num=2, base_model="mobilenet")
    model_logger.debug(f"Loading model: {model_rise_path}")
    model_rise = load_model(model_rise_path, class_num=2, base_model="resnet152_600")
    model_logger.debug(f"Loading model: {model_shalvar_path}")
    model_shalvar = load_model(model_shalvar_path, class_num=7, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_mnist_path}")
    model_mnist = load_model(model_mnist_path, class_num=2, base_model="mobilenet-v2")
    model_logger.debug(f"Loading model: {model_tarh_shalvar_path}")
    model_tarh_shalvar = load_model(model_tarh_shalvar_path, class_num=5, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_skirt_pants_path}")
    model_skirt_pants = load_model(model_skirt_pants_path, class_num=2, base_model="resnet101")
    model_logger.debug(f"Loading model: {model_yaghe_path}")
    model_yaghe = load_model(model_yaghe_path, class_num=5, base_model="mobilenet-v2-softmax")
    model_logger.debug(f"Loading model: {model_yolo_path}")
    model_yolo = load_model(model_yolo_path, class_num=2, base_model="yolo")
    model_logger.debug(f"Loading model: {model_body_type_path}")
    model_body_type = load_model(model_body_type_path, class_num=3, base_model="bodytype")

    total_time = time.time() - start_time
    model_logger.info(f"Successfully loaded all men's clothing models in {total_time:.2f} seconds")
except Exception as e:
    model_logger.error(f"Failed to load men's clothing models: {str(e)}")
    raise


def process_clothing_image(img_path):
    """
    Processes a clothing image to predict various attributes such as color tone,
    clothing type (paintane), sleeve type (astin), pattern, neckline (yaghe),
    rise, shalvar type, shalvar pattern, and skirt/pants classification.

    Args:
        img_path (str): The path to the clothing image file.

    Returns:
        dict: A dictionary containing the prediction results for various clothing attributes.
              The dictionary includes the following keys:
              - "color_tone": The dominant color tone of the image.
              - "mnist_prediction": Prediction from the MNIST model (Under/Over).
              - "paintane": Predicted clothing type (mbalatane/mpayintane).
              - "astin" (conditional): Sleeve type prediction if paintane is "mbalatane".
              - "pattern" (conditional): Pattern prediction if paintane is "mbalatane".
              - "yaghe" (conditional): Neckline prediction if paintane is "mbalatane".
              - "rise" (conditional): Rise prediction if paintane is "mpayintane".
              - "shalvar" (conditional): Shalvar type prediction if paintane is "mpayintane".
              - "tarh_shalvar" (conditional): Shalvar pattern prediction if paintane is "mpayintane".
              - "skirt_and_pants" (conditional): Skirt/pants classification if paintane is "mpayintane".
    """
    # 1. Image Loading and Preprocessing
    img = cv2.imread(img_path)

    response = {
        "ok": True,
        "data": {},
        "error": None
    }

    # Validate image was loaded successfully
    if img is None:
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
        detections = detect_clothing(img_path)
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
        # If the validation model fails, continue processing but log the error
    
    model_logger.info(f"Processing image: {img_path}")
    results = {}

    # 2. General Predictions (Independent of clothing type)
    model_logger.info("Start color_tone prediction")
    try:
        results["color_tone"] = get_color_tone(img_path)
        model_logger.info(f"Color tone prediction result: {results.get('color_tone')}")
    except Exception as e:
        model_logger.error(f"Color tone prediction failed: {e}")
        results["color_tone"] = None

    model_logger.info("Start paintane prediction")
    try:
        results["paintane"] = predict_class(img, model=model_paintane, class_names=["mbalatane", 'mpayintane'], reso=224, model_name="paintane")
        model_logger.info(f"Paintane prediction result: {results.get('paintane')}")
    except Exception as e:
        model_logger.error(f"Paintane prediction failed: {e}")
        results["paintane"] = None

    # 3. Conditional Predictions based on 'paintane' (Clothing Type)
    # Upper body clothing
    if results["paintane"] == "mbalatane":
        model_logger.info("Detected 'mbalatane' (upper body clothing). Proceeding with upper body predictions.")
        try:
            crop_image_astin, crop_image_yaghe = yolo_predict_crop(model=model_yolo, image_path=img_path, image=img)
            model_logger.info(f"YOLO detection completed. astin crop: {crop_image_astin is not None}, yaghe crop: {crop_image_yaghe is not None}")
        except Exception as e:
            model_logger.error(f"YOLO crop failed: {e}")
            crop_image_astin, crop_image_yaghe = None, None

        try:
            results["mnist_prediction"] = predict_class(img, model=model_mnist, class_names=["Under", "Over"], reso=224, model_name="mnist")
            model_logger.info(f"MNIST prediction result: {results.get('mnist_prediction')}")
        except Exception as e:
            model_logger.error(f"MNIST prediction failed: {e}")
            results["mnist_prediction"] = None

        # Astin prediction only if crop is not None
        if crop_image_astin is not None:
            try:
                results["astin"] = predict_class(crop_image_astin, model=model_astin, class_names=["longsleeve", "shortsleeve", "sleeveless"], reso=300, model_name="astin")
                model_logger.info(f"Astin prediction result: {results.get('astin')}")
            except Exception as e:
                model_logger.error(f"Astin prediction failed: {e}")
                results["astin"] = None
        else:
            model_logger.error("Astin crop is None, skipping astin prediction.")
            results["astin"] = None

        try:
            results["pattern"] = predict_class(img, model=model_patern, class_names=["amudi", "dorosht", "ofoghi", "riz", "sade"], reso=300, model_name="pattern")
            model_logger.info(f"Pattern prediction result: {results.get('pattern')}")
        except Exception as e:
            model_logger.error(f"Pattern prediction failed: {e}")
            results["pattern"] = None

        # Yaghe prediction only if crop is not None
        if crop_image_yaghe is not None:
            try:
                results["yaghe"] = predict_class(crop_image_yaghe, model=model_yaghe, class_names=["classic", "hoodie", "round", "turtleneck", "V_neck"], reso=300, model_name="yaghe")
                model_logger.info(f"Yaghe prediction result: {results.get('yaghe')}")
            except Exception as e:
                model_logger.error(f"Yaghe prediction failed: {e}")
                results["yaghe"] = None
        else:
            model_logger.error("Yaghe crop is None, skipping yaghe prediction.")
            results["yaghe"] = None

    # Lower body clothing
    elif results["paintane"] == "mpayintane":
        model_logger.info("Detected 'mpayintane' (lower body clothing). Proceeding with lower body predictions.")
        try:
            results["rise"] = predict_class(img, model=model_rise, class_names=["highrise", "lowrise"], reso=300, model_name="rise")
            model_logger.info(f"Rise prediction result: {results.get('rise')}")
        except Exception as e:
            model_logger.error(f"Rise prediction failed: {e}")
            results["rise"] = None

        try:
            results["shalvar"] = predict_class(img, model=model_shalvar, class_names=["mbaggy", "mcargo", "mcargoshorts", "mmom", "mshorts", "mslimfit", "mstraight"], reso=300, model_name="noe shalvar")
            model_logger.info(f"Shalvar prediction result: {results.get('shalvar')}")
        except Exception as e:
            model_logger.error(f"Shalvar prediction failed: {e}")
            results["shalvar"] = None

        try:
            results["tarh_shalvar"] = predict_class(img, model=model_tarh_shalvar, class_names=["mpamudi", "mpdorosht", "mpofoghi", "mpriz", "mpsade"], reso=300, model_name="tarhshalvar")
            model_logger.info(f"Tarh shalvar prediction result: {results.get('tarh_shalvar')}")
        except Exception as e:
            model_logger.error(f"Tarh shalvar prediction failed: {e}")
            results["tarh_shalvar"] = None

        try:
            results["skirt_and_pants"] = predict_class(img, model=model_skirt_pants, class_names=["pants", "skirt"], reso=300, model_name="skirt and pants")
            model_logger.info(f"Skirt and pants prediction result: {results.get('skirt_and_pants')}")
        except Exception as e:
            model_logger.error(f"Skirt and pants prediction failed: {e}")
            results["skirt_and_pants"] = None

    model_logger.info(f"Returning results: {results}")
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


def get_man_body_type(image_path):

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
        img=image,
        model=model_body_type,
        class_names=['0', '1', '2'],
        reso=300,
        model_name="bodytype"
    )
    
    response["data"]["body_type"] = body_type
    return response



def test_model_astin(image_path="../../image/sample_astin.jpg"):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_astin,
                           class_names=["longsleeve", "shortsleeve", "sleeveless"], reso=300,
                           model_name="astin")
    print(f"Astin Prediction: {result}")


def test_model_patern(image_path="../../image/sample_patern.jpg"):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_patern,
                           class_names=["amudi", "dorosht", "ofoghi", "riz", "sade"], reso=300,
                           model_name="pattern")
    print(f"Patern Prediction: {result}")


def test_model_paintane(image_path="../../image/sample_paintane.jpg"):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_paintane,
                           class_names=["mbalatane", 'mpayintane'], reso=224,
                           model_name="paintane")
    print(f"Paintane Prediction: {result}")


def test_model_rise(image_path="../../image/sample_rise.jpg"):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_rise,
                           class_names=["highrise", "lowrise"], reso=300,
                           model_name="rise")
    print(f"Rise Prediction: {result}")


def test_model_shalvar(image_path="../../image/sample_shalvar.jpg"):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_shalvar,
                           class_names=["mbaggy", "mcargo", "mcargoshorts", "mmom", "mshorts",
                                        "mslimfit", "mstraight"], reso=300, model_name="noe shalvar")
    print(f"Shalvar Prediction: {result}")


def test_model_mnist(image_path="../../image/sample_mnist.jpg"):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_mnist, class_names=["Under", "Over"], reso=224, model_name="mnist")
    print(f"MNIST Prediction: {result}")


def test_model_tarh_shalvar(image_path="../../image/sample_tarh_shalvar.jpg"):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_tarh_shalvar,
                           class_names=["mpamudi", "mpdorosht", "mpofoghi", "mpriz", "mpsade"],
                           reso=300, model_name="tarhshalvar")
    print(f"Tarh Shalvar Prediction: {result}")


def test_model_skirt_pants(image_path="../../image/sample_skirt_pants.jpg"):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_skirt_pants,
                           class_names=["pants", "skirt"], reso=300, model_name="skirt and pants")
    print(f"Skirt and Pants Prediction: {result}")


def test_model_yaghe(image_path="../../image/sample_yaghe.jpg"):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_yaghe,
                           class_names=["classic", "hoodie", "round", "turtleneck", "V_neck"], reso=300,
                           model_name="yaghe")
    print(f"Yaghe Prediction: {result}")
