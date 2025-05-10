"""
test 
"""
from model_handler_service.core.loaders import load_model, predict_class, yolo_predict_crop
from model_handler_service.color import get_color_tone
from model_handler_service.load_and_predict_woman import model_astin,model_patern,model_paintane,model_rise,model_shalvar,model_tarh_shalvar
from model_handler_service.load_and_predict_woman import model_skirt_pants,model_yaghe,model_skirt_print,model_skirt_type,model_mnist,model_body_type
from model_handler_service.load_and_predict_woman import model_yolo,model_balted,model_cowl,model_empire,model_loose,model_peplum,model_wrap
import cv2

def test_model_astin(image_path):
    # Crop astin and yaghe
    crop_image_astin, _ = yolo_predict_crop(model=model_yolo, image_path=image_path)
    result = predict_class(crop_image_astin, model=model_astin,
                           class_names=['bottompuffy', "fhalfsleeve", "flongsleeve", "fshortsleeve", "fsleeveless", "toppuffy"], reso=300,
                           model_name="astin")
    print(f"Astin Prediction: {result}")


def test_model_patern(image_path):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_patern,
                           class_names=["dorosht", "rahrahamudi", "rahrahofoghi", "riz", "sade"], reso=300,
                           model_name="pattern")
    print(f"Patern Prediction: {result}")


def test_model_paintane(image_path):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_paintane,
                           class_names=["fbalatane", "fpaintane", "ftamamtane"], reso=224,
                           model_name="paintane")
    print(f"Paintane Prediction: {result}")


def test_model_rise(image_path):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_rise,
                           class_names=["highrise", "lowrise"], reso=300,
                           model_name="rise")
    print(f"Rise Prediction: {result}")


def test_model_shalvar(image_path):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_shalvar,
                           class_names=["wbaggy", "wbootcut", "wcargo", "wcargoshorts", "wmom", "wshorts", "wskinny", "wstraight"], reso=300, model_name="noeshalvar")
    print(f"Shalvar Prediction: {result}")


def test_model_mnist(image_path):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_mnist, class_names=["Under", "Over"], reso=224, model_name="mnist")
    print(f"MNIST Prediction: {result}")


def test_model_tarh_shalvar(image_path):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_tarh_shalvar,
                           class_names=["wpamudi", "wpdorosht", "wpofoghi", "wpriz", "wpsade"],
                           reso=300, model_name="tarhshalvar")
    print(f"Tarh Shalvar Prediction: {result}")


def test_model_skirt_pants(image_path):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_skirt_pants,
                           class_names=["pants", "skirt"], reso=300, model_name="skirt and pants")
    print(f"Skirt and Pants Prediction: {result}")


def test_model_yaghe(image_path):
    # Crop astin and yaghe
    _, crop_image_yaghe = yolo_predict_crop(model=model_yolo, image_path=image_path)
    result = predict_class(crop_image_yaghe, model=model_yaghe,
                           class_names=["boatneck", "classic", "halter", "hoodie", "of_the_shoulder", "one_shoulder", 
                                        "round", "squer", "sweatheart", 'turtleneck', "v_neck"], reso=300,
                           model_name="yaghe")
    print(f"Yaghe Prediction: {result}")

def test_model_skirt_print(image_path):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_skirt_print, 
                           class_names=["skirtamudi", "skirtdorosht", "skirtofoghi", "skirtriz", "skirtsade"], 
                           reso=300, model_name="skirt and pants")
    print(f"Skirt Print Prediction: {result}")

def test_model_skirt_type(image_path):
    sample_image = cv2.imread(image_path)
    result = predict_class(sample_image, model=model_skirt_type, 
                           class_names=["alineskirt", "balloonskirt", "mermaidskirt", "miniskirt", "pencilskirt", "shortaskirt", "wrapskirt"], 
                           reso=300, model_name="model_skirt_type")
    print(f"Skirt Print Prediction: {result}")


#test_model_astin(r"D:\chibepoosham\model_handler_service\src\model_handler_service\balatane_men.jpg")



