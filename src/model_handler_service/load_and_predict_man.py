import os
import cv2
from model_handler_service.core.loaders import load_model, predict_class, yolo_predict_crop
from model_handler_service.core.processing import preprocess_image_for_bodytype
from model_handler_service.core.validations import validate_human_image
# from model_handler_service.color import get_color_tone


# Define model paths
# For Docker
# model_astin_path = '/var/www/deploy/models/astin/astinman.h5'
# model_patern_path = '/var/www/deploy/models/patern/petternman.h5'
# model_paintane_path = '/var/www/deploy/models/paintane/mard.h5'
# model_rise_path = '/var/www/deploy/models/rise/riseeeeef.h5'
# model_shalvar_path = '/var/www/deploy/models/shalvar/menpants.h5'
# model_mnist_path = '/var/www/deploy/models/fasionmnist/mnist.h5'
# model_tarh_shalvar_path = '/var/www/deploy/models/tarh_shalvar/mmpantsprint.h5'
# model_skirt_pants_path = '/var/www/deploy/models/skirt_pants/skirt_pants.h5'
# model_yaghe_path = '/var/www/deploy/models/yaghe/neckline_classifier_mobilenet.h5'
# model_yolo = '/var/www/deploy/models/yolo/best.pt'
# model_body_type_path = '/var/www/deploy/models/body_type/models/model_body_type.pth'

# For Local
base_path = os.path.dirname(__file__)
model_astin_path = os.path.join(base_path, '../../models/astin/astinman.h5')
model_patern_path = os.path.join(base_path, '../../models/pattern/petternman.h5')
model_paintane_path = os.path.join(base_path, '../../models/paintane/mard.h5')
model_rise_path = os.path.join(base_path, '../../models/rise/riseeeeef.h5')
model_shalvar_path = os.path.join(base_path, '../../models/shalvar/menpants.h5')
model_mnist_path = os.path.join(base_path, '../../models/under_over/under_over_mobilenet_final.h5')
model_tarh_shalvar_path = os.path.join(base_path, '../../models/tarh_shalvar/mmpantsprint.h5')
model_skirt_pants_path = os.path.join(base_path, '../../models/skirt_pants/skirt_pants.h5')
model_yaghe_path = os.path.join(base_path, '../../models/yaghe/neckline_classifier_mobilenet.h5')
model_yolo_path = os.path.join(base_path, '../../models/yolo/best.pt')
model_body_type_path = os.path.join(base_path, '../../models/body_type/model_body_type.pth')

# Load models globally
model_astin = load_model(model_astin_path, class_num=3, base_model="resnet101")
model_patern = load_model(model_patern_path, class_num=5, base_model="resnet101")
model_paintane = load_model(model_paintane_path, class_num=2, base_model="mobilenet")
model_rise = load_model(model_rise_path, class_num=2, base_model="resnet152_600")
model_shalvar = load_model(model_shalvar_path, class_num=7, base_model="resnet101")
model_mnist = load_model(model_mnist_path, class_num=2, base_model="mobilenet-v2")
model_tarh_shalvar = load_model(model_tarh_shalvar_path, class_num=5, base_model="resnet101")
model_skirt_pants = load_model(model_skirt_pants_path, class_num=2, base_model="resnet101")
model_yaghe = load_model(model_yaghe_path, class_num=5, base_model="mobilenet-v2-softmax")
model_yolo = load_model(model_yolo_path, class_num=2, base_model="yolo")
model_body_type = load_model(model_body_type_path, class_num=3, base_model="bodytype")


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
    
    print(f"Processing image: {img_path}")
    results = {}

    # 2. General Predictions (Independent of clothing type)
    # Predict color tone
    # results["color_tone"] = get_color_tone(img)
    print(f"Color tone prediction: {results.get('color_tone')}")

    # Predict clothing type (paintane or balatane)
    results["paintane"] = predict_class(img, model=model_paintane, class_names=["mbalatane", 'mpayintane'], reso=224, model_name="paintane")
    print(f"Paintane prediction: {results.get('paintane')}")

    # 3. Conditional Predictions based on 'paintane' (Clothing Type)
    # Upper body clothing
    if results["paintane"] == "mbalatane":
        print("Detected 'mbalatane' (upper body clothing). Proceeding with upper body predictions.")

        # Crop astin and yaghe
        crop_image_astin, crop_image_yaghe = yolo_predict_crop(model=model_yolo, image_path=img_path, image=img)
        print(f"YOLO detection completed. astin crop: {crop_image_astin is not None}, yaghe crop: {crop_image_yaghe is not None}")

        # MNIST prediction for under/over clothing
        results["mnist_prediction"] = predict_class(img, model=model_mnist, class_names=["Under", "Over"], reso=224, model_name="mnist")
        print(f"MNIST prediction: {results.get('mnist_prediction')}")

        # Predict sleeve type (astin)
        results["astin"] = predict_class(crop_image_astin, model=model_astin,
                                         class_names=["longsleeve", "shortsleeve", "sleeveless"], reso=300,
                                         model_name="astin")
        print(f"Astin prediction: {results.get('astin')}")

        # Predict pattern
        results["pattern"] = predict_class(img, model=model_patern,
                                           class_names=["amudi", "dorosht", "ofoghi", "riz", "sade"], reso=300,
                                           model_name="pattern")
        print(f"Pattern prediction: {results.get('pattern')}")

        # Predict neckline (yaghe)
        results["yaghe"] = predict_class(crop_image_yaghe, model=model_yaghe,
                                         class_names=["classic", "hoodie", "round", "turtleneck", "V_neck"], reso=300,
                                         model_name="yaghe")
        print(f"Yaghe prediction: {results.get('yaghe')}")

    # Lower body clothing
    elif results["paintane"] == "mpayintane":
        print("Detected 'mpayintane' (lower body clothing). Proceeding with lower body predictions.")

        # Predict rise
        results["rise"] = predict_class(img, model=model_rise, class_names=["highrise", "lowrise"], reso=300,
                                        model_name="rise")
        print(f"Rise prediction: {results.get('rise')}")

        # Predict shalvar type
        results["shalvar"] = predict_class(img, model=model_shalvar,
                                           class_names=["mbaggy", "mcargo", "mcargoshorts", "mmom", "mshorts",
                                                        "mslimfit", "mstraight"], reso=300, model_name="noe shalvar")
        print(f"Shalvar prediction: {results.get('shalvar')}")

        # Predict shalvar pattern
        results["tarh_shalvar"] = predict_class(img, model=model_tarh_shalvar,
                                                class_names=["mpamudi", "mpdorosht", "mpofoghi", "mpriz", "mpsade"],
                                                reso=300, model_name="tarhshalvar")
        print(f"Tarh shalvar prediction: {results.get('tarh_shalvar')}")

        # Predict skirt or pants
        results["skirt_and_pants"] = predict_class(img, model=model_skirt_pants, class_names=["pants", "skirt"],
                                                   reso=300, model_name="skirt and pants")
        print(f"Skirt and pants prediction: {results.get('skirt_and_pants')}")
    
    print("Returning results:")
    print(results)
    response["data"] = results
    return response


def get_man_body_type(image_path):

    response = {
        "ok": True,
        "data": {
            "body_type": "string",
        },
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
    human_validation_errors = validate_human_image()
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
        class_names=['men_inverted_triangle', 'men_oval', 'men_rectangle'],
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
