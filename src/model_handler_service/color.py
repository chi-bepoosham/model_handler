import cv2
import numpy as np
from sklearn.cluster import KMeans
from colorsys import rgb_to_hls


# for Docker
# model_mnist_path = '/var/www/deploy/models/under_over/under_over_mobilenet_final.h5'
# model_sam = '/var/www/deploy/models/under_over/sam_vit_b_01ec64.pth'

# For Local
# base_path = os.path.dirname(__file__)
# model_detect_clothes = os.path.join(base_path, '../../models/validations/clothes/detect_clothes.pt')
# model_sam = os.path.join(base_path, '../../models/validations/clothes/sam_vit_b_01ec64.pth')

# model = YOLO(model_detect_clothes)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# sam = sam_model_registry["vit_h"](checkpoint=model_sam).to(device)
# mask_generator = SamAutomaticMaskGenerator(sam)


def detect_clothing(image_path):
    
    results = model(image_path)
    
    detections = []
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            detections.append((x1, y1, x2, y2))
    
    return detections


def segment_clothing(image_path, bbox):

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    x1, y1, x2, y2 = bbox
    cropped_img = image_rgb[y1:y2, x1:x2]

    masks = mask_generator.generate(cropped_img)

    if len(masks) == 0:
        print("⚠️ No mask found! Using bounding box as mask.")
        return cropped_img  

    largest_mask = max(masks, key=lambda x: np.sum(x["segmentation"]))
    mask = largest_mask["segmentation"]

    segmented_img = np.zeros_like(cropped_img)
    segmented_img[mask] = cropped_img[mask]

    return segmented_img


def extract_dominant_color(image, k=3):
    if image is None or image.size == 0:
        print("⚠️ Segmented image is invalid!")
        return np.array([0, 0, 0])  

    image = cv2.resize(image, (100, 100))
    image = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(image)
    
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

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
    segmented_image = segment_clothing(image_path, bbox)

    dominant_color = extract_dominant_color(segmented_image)

    if np.all(dominant_color == 0):
        print("⚠️ No dominant color detected!")
        return

    color_category = classify_color(dominant_color)

    print(f"🚀 Main Color: {dominant_color}  🎨 Classification: {color_category}")

    return color_category
