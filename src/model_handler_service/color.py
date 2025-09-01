
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from colorsys import rgb_to_hls
from sklearn.cluster import KMeans
from model_handler_service.core.logger import model_logger
from model_handler_service.core.config import config



# Initialize device (force CPU)
device = torch.device("cpu")
model_logger.info(f"Using device for color analysis: {device}")

# Define model paths using the MODEL_PATH from config
# This will use the path defined in the .env file
model_path = str(config.get_model_file_path('validations/clothes/detect_clothes.pt'))
sam_checkpoint = str(config.get_model_file_path('color/sam_vit_b_01ec64.pth'))

# Load models
model_logger.info("Starting to load color analysis models")
start_time = time.time()

try:
    # Load YOLO model for clothing detection
    model_logger.info(f"Loading YOLO model from: {os.path.basename(model_path)}")
    model = YOLO(model_path)
    
    # Load SAM model for segmentation
    model_type="vit_b"
    model_logger.info(f"Loading SAM model from: {os.path.basename(sam_checkpoint)}")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    
    # Initialize mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    total_time = time.time() - start_time
    model_logger.info(f"Successfully loaded all color analysis models in {total_time:.2f} seconds")
except Exception as e:
    model_logger.error(f"Failed to load color analysis models: {str(e)}")
    raise


def detect_clothing(image_path):
    """
    Run YOLO validations model on the image and return bounding boxes
    that pass a minimum confidence threshold.

    Returns list of (x1, y1, x2, y2).
    """
    # minimum confidence to accept a detection (tweakable)
    min_conf = float(os.getenv("VALIDATIONS_MIN_CONF", 0.35))

    results = model(image_path)

    detections = []
    for r in results:
        # r.boxes.conf contains confidences for each box when available
        confs = getattr(r.boxes, "conf", None)
        xyxys = list(r.boxes.xyxy)
        if confs is None:
            # fallback: accept boxes if no conf info, but log a warning
            model_logger.warning("YOLO results have no confidences; accepting all boxes")
            for box in xyxys:
                x1, y1, x2, y2 = map(int, box)
                detections.append((x1, y1, x2, y2))
        else:
            for box, conf in zip(xyxys, confs):
                try:
                    c = float(conf)
                except Exception:
                    c = 0.0
                if c >= min_conf:
                    x1, y1, x2, y2 = map(int, box)
                    detections.append((x1, y1, x2, y2))

    model_logger.info(f"Validations YOLO detections: {len(detections)} (min_conf={min_conf}) for {os.path.basename(image_path)}")
    return detections


def segment_clothing(image_path, bbox):

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    x1, y1, x2, y2 = bbox
    cropped_img = image_rgb[y1:y2, x1:x2]

    masks = mask_generator.generate(cropped_img)

    if len(masks) == 0:
        print("⚠️ No mask found! Using bounding box as mask.")
        return cropped_img, None  

    largest_mask = max(masks, key=lambda x: np.sum(x["segmentation"]))
    mask = largest_mask["segmentation"]

    segmented_img = np.zeros_like(cropped_img)
    segmented_img[mask] = cropped_img[mask]

    return segmented_img, cropped_img 


def extract_dominant_color(image, k=3):
    if image is None or image.size == 0:
        print("⚠️ Segmented image is invalid!")
        return np.array([0, 0, 0])  

    model_logger.info(f"Extracting dominant color from image of shape {image.shape}")

    image = cv2.resize(image, (100, 100))
    image = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(image)

    labels = kmeans.labels_
    bin_counts = np.bincount(labels)
    dominant_label = np.argmax(bin_counts)

    model_logger.info(f"KMeans labels: {labels}")
    model_logger.info(f"Bin counts: {bin_counts}")
    model_logger.info(f"Dominant label: {dominant_label}")

    dominant_color = kmeans.cluster_centers_[dominant_label]

    model_logger.info(f"Dominant color: {dominant_color}")

    return dominant_color.astype(int)


def classify_color(dominant_color):

    r, g, b = dominant_color
    h, l, s = rgb_to_hls(r/255.0, g/255.0, b/255.0)

    if l > 0.5: 
        category = "light_bright" if s > 0.5 else "light_muted"
    else: 
        category = "dark_bright" if s > 0.5 else "dark_muted"

    return category


def get_color_tone(image_path):
    """
    Analyzes the lightness and saturation of the dominant color in an image.

    Parameters:
    - image_path: image path

    Returns:
    - tone
    """

    detections = detect_clothing(image_path)

    if not detections:
        print("⚠️ No clothing detected!")
        return

    bbox = detections[0]
    _, cropped_img = segment_clothing(image_path, bbox)

    dominant_color = extract_dominant_color(cropped_img)

    if np.all(dominant_color == 0):
        print("⚠️ No dominant color detected!")
        return

    color_category = classify_color(dominant_color)

    return color_category
