import os
import cv2
import time
from model_handler_service.core.loaders import load_model, predict_class, yolo_predict_crop
from model_handler_service.core.config import config
from model_handler_service.core.logger import model_logger
from model_handler_service.color import get_color_tone
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
    model_astin = load_model(model_astin_path, class_num=6, base_model="resnet101")
    model_patern = load_model(model_patern_path, class_num=5, base_model="resnet101")
    model_paintane = load_model(model_paintane_path, class_num=3, base_model="mobilenet-v2-pt")
    model_rise = load_model(model_rise_path, class_num=2, base_model="resnet152_600")
    model_shalvar = load_model(model_shalvar_path, class_num=8, base_model="resnet101")
    model_tarh_shalvar = load_model(model_tarh_shalvar_path, class_num=5, base_model="resnet101")
    model_skirt_pants = load_model(model_skirt_pants_path, class_num=2, base_model="resnet101")
    model_yaghe = load_model(model_yaghe_path, class_num=11, base_model="resnet101")
    model_skirt_print = load_model(model_skirt_print_path, class_num=5, base_model="resnet101_30_unit")
    model_skirt_type = load_model(model_skirt_type_path, class_num=7, base_model="resnet101_30_unit")
    model_mnist = load_model(model_mnist_path, class_num=2, base_model="mobilenet-v2")
    
    model_logger.info("Loading PyTorch body type model")
    model_body_type = torch.load(model_body_type_path, map_location=torch.device('cpu'))  # Load PyTorch model body type (arian)
    
    model_yolo = load_model(model_yolo_path, class_num=2, base_model="yolo")
    model_balted = load_model(model_balted_path, class_num=2, base_model="resnet101")
    model_cowl = load_model(model_cowl_path, class_num=2, base_model="resnet101")
    model_empire = load_model(model_empire_path, class_num=2, base_model="resnet101")
    model_loose = load_model(model_loose_path, class_num=2, base_model="resnet101")
    model_peplum = load_model(model_peplum_path, class_num=2, base_model="resnet101")
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

    print(f"Processing image: {image_path}")
    results = {}

    # 2. General Predictions (Independent of clothing type)
    # Predict color tone
    results["color_tone"] = get_color_tone(image_path)
    print(f"Color tone prediction: {results.get('color_tone')}")

    # Predict clothing type (paintane)
    results["paintane"] = predict_class(image, model=model_paintane, 
                                       class_names=["fbalatane", "fpaintane", "ftamamtane"], 
                                       reso=224, model_name="paintane")
    print(f"Paintane prediction: {results.get('paintane')}")

    # 3. Conditional Predictions based on 'paintane' (Clothing Type)
    # Upper body or full body clothing
    if results["paintane"] in ["fbalatane", "ftamamtane"]:
        print("Detected 'fbalatane' or 'ftamamtane'. Proceeding with upper body predictions.")

        # MNIST prediction for under/over clothing
        results["mnist_prediction"] = predict_class(image, model=model_mnist, class_names=["Under", "Over"], reso=224, model_name="mnist")
        print(f"MNIST prediction: {results.get('mnist_prediction')}")

        
        # Crop astin and yaghe
        crop_image_astin, crop_image_yaghe = yolo_predict_crop(model=model_yolo, image_path=image_path)
        print(f"YOLO detection completed. astin crop: {crop_image_astin is not None}, yaghe crop: {crop_image_yaghe is not None}")
        
        # Predict sleeve type (astin)
        results["astin"] = predict_class(crop_image_astin, model=model_astin, 
                                        class_names=['bottompuffy', "fhalfsleeve", "flongsleeve", "fshortsleeve", "fsleeveless", "toppuffy"], 
                                        reso=300, model_name="astin")
        print(f"Astin prediction: {results.get('astin')}")
        
        # Predict pattern
        results["pattern"] = predict_class(image, model=model_patern, 
                                          class_names=["dorosht", "rahrahamudi", "rahrahofoghi", "riz", "sade"], 
                                          reso=300, model_name="pattern")
        print(f"Pattern prediction: {results.get('pattern')}")
        
        # Predict neckline (yaghe)
        results["yaghe"] = predict_class(crop_image_yaghe, model=model_yaghe, 
                                        class_names=["boatneck", "classic", "halter", "hoodie", "of_the_shoulder", "one_shoulder", "round", "squer", "sweatheart", 'turtleneck', "v_neck"], 
                                        reso=300, model_name="yaghe")
        print(f"Yaghe prediction: {results.get('yaghe')}")

    # Lower body or full body clothing
    if results["paintane"] in ["fpaintane", "ftamamtane"]:
        print("Detected 'fpaintane' or 'ftamamtane'. Proceeding with lower body predictions.")
        
        # Predict rise
        results["rise"] = predict_class(image, model=model_rise, 
                                       class_names=["highrise", "lowrise"], 
                                       reso=300, model_name="rise")
        print(f"Rise prediction: {results.get('rise')}")
        
        # Predict shalvar type
        results["shalvar"] = predict_class(image, model=model_shalvar, 
                                          class_names=["wbaggy", "wbootcut", "wcargo", "wcargoshorts", "wmom", "wshorts", "wskinny", "wstraight"], 
                                          reso=300, model_name="noeshalvar")
        print(f"Shalvar prediction: {results.get('shalvar')}")
        
        # Predict shalvar pattern
        results["tarh_shalvar"] = predict_class(image, model=model_tarh_shalvar, 
                                               class_names=["wpamudi", "wpdorosht", "wpofoghi", "wpriz", "wpsade"], 
                                               reso=300, model_name="tarhshalvar")
        print(f"Tarh shalvar prediction: {results.get('tarh_shalvar')}")
        
        # Predict skirt or pants
        results["skirt_and_pants"] = predict_class(image, model=model_skirt_pants, 
                                                  class_names=["pants", "skirt"], 
                                                  reso=300, model_name="skirt and pants")
        print(f"Skirt and pants prediction: {results.get('skirt_and_pants')}")
        
        # Predict skirt print
        results["skirt_print"] = predict_class(image, model=model_skirt_print, 
                                              class_names=["skirtamudi", "skirtdorosht", "skirtofoghi", "skirtriz", "skirtsade"], 
                                              reso=300, model_name="skirt and pants")
        print(f"Skirt print prediction: {results.get('skirt_print')}")
        
        # Predict skirt type
        results["skirt_type"] = predict_class(image, model=model_skirt_type, 
                                             class_names=["alineskirt", "balloonskirt", "mermaidskirt", "miniskirt", "pencilskirt", "shortaskirt", "wrapskirt"], 
                                             reso=300, model_name="model_skirt_type")
        print(f"Skirt type prediction: {results.get('skirt_type')}")
    
    response["data"] = results
    return response


def get_body_type_female(image_path):
    image = cv2.imread(image_path)
    
    response = {
        "ok": True,
        "data": {
            "body_type": "string",
        },
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

    # TODO: Add actual human validation logic here
    # For now returning mock validation
    human_validation_errors = []
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
        class_names=["11", "21", "31", "41", "51"],
        reso=300,
        model_name="bodytype"
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
    image = cv2.imread(image_path)
    
    # Perform predictions
    results = {}
    
    # Predict balted
    results["balted"] = predict_class(image, model=model_balted, 
                                     class_names=["balted", "notbalted"], 
                                     reso=300, model_name="balted")
    print(f"Balted prediction: {results.get('balted')}")
    
    # Predict cowl
    results["cowl"] = predict_class(image, model=model_cowl, 
                                   class_names=["cowl", "notcowl"], 
                                   reso=300, model_name="cowl")
    print(f"Cowl prediction: {results.get('cowl')}")
    
    # Predict empire
    results["empire"] = predict_class(image, model=model_empire, 
                                     class_names=["empire", "notempire"], 
                                     reso=300, model_name="empire")
    print(f"Empire prediction: {results.get('empire')}")
    
    # Predict loose
    results["loose"] = predict_class(image, model=model_loose, 
                                    class_names=["losse", "snatched"], 
                                    reso=300, model_name="loose")
    print(f"Loose prediction: {results.get('loose')}")
    
    # Predict wrap
    results["wrap"] = predict_class(image, model=model_wrap, 
                                   class_names=["notwrap", "wrap"], 
                                   reso=300, model_name="wrap")
    print(f"Wrap prediction: {results.get('wrap')}")
    
    # Predict peplum
    results["peplum"] = predict_class(image, model=model_peplum, 
                                     class_names=["notpeplum", "peplum"], 
                                     reso=300, model_name="peplum")
    print(f"Peplum prediction: {results.get('peplum')}")
    
    print("Returning 6 model results:")
    print(results)
    
    return results
