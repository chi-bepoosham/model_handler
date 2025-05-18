from flask import Flask, request, jsonify
import requests
import time
import random
import os


from model_handler_service.load_and_predict_man import process_clothing_image, get_man_body_type
from model_handler_service.load_and_predict_woman import process_woman_clothing_image, process_six_model_predictions, get_body_type_female
from model_handler_service.core.config import config
from model_handler_service.core.logger import model_logger

# Temporary directory for storing images
TEMP_IMAGES_DIR = config.get_temp_dir()

app = Flask(__name__)


def download_image(image_url):
    """Downloads an image from the given URL and saves it in the temporary directory."""
    try:
        filename = image_url.split('/')[-1]
        # Check if the filename has a common image extension
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        has_extension = any(filename.lower().endswith(ext) for ext in image_extensions)

        # If no common extension is found, add a default one (e.g., .jpg)
        if not has_extension:
            filename += '.jpg'
        img_data = requests.get(image_url).content
        img_name = os.path.join(TEMP_IMAGES_DIR, f"{int(time.time())}{random.randrange(100, 999)}-temp-{filename}")

        with open(img_name, 'wb') as handler:
            handler.write(img_data)
        
        return img_name
    except Exception as e:
        return None


@app.route("/healthcheck", methods=['GET'])
def health_check():
    """Health check endpoint to verify that the API is running."""
    return jsonify({"ok": True, "data": {"status": "API is running"}, "error": None}), 200


@app.route('/clothing', methods=['POST'])
def classify_clothing():
    """API endpoint for clothing classification."""
    data = request.json
    model_logger.info("Received request for clothing classification.")

    # Extract input parameters
    gender = data.get('gender')
    image_url = data.get('image_url')

    # Validate input data
    if not image_url or gender not in [0, 1, "0", "1"]:
        model_logger.warning("Invalid input data - missing image URL or invalid gender value.")
        return jsonify({
            "ok": False,
            "data": None,
            "error": {
                "code": "INVALID_INPUT",
                "message": "Invalid input data - missing image URL or invalid gender value"
            }
        }), 400

    # Download image
    img_path = download_image(image_url)
    if not img_path:
        model_logger.error("Failed to download image from provided URL.")
        return jsonify({
            "ok": False,
            "data": None,
            "error": {
                "code": "IMAGE_DOWNLOAD_FAILED", 
                "message": "Failed to download image from provided URL"
            }
        }), 500

    # Process image based on gender
    if str(gender) == "1":  # Male
        model_logger.info("Processing clothing image for male.")
        process_data = process_clothing_image(img_path)
    else:  # Female
        model_logger.info("Processing clothing image for female.")
        process_data = process_woman_clothing_image(img_path)
        
        # Check if the clothing type is not "fpaintane" (not lower body)
        if process_data.get('paintane') != "fpaintane":
            six_model_results = process_six_model_predictions(img_path)
            process_data["data"].update(six_model_results)  # Merge both results

    model_logger.info("Clothing classification completed successfully.")
    return jsonify(process_data)


@app.route('/bodytype', methods=['POST'])
def classify_bodytype():
    """API endpoint for body type classification."""
    data = request.json
    model_logger.info("Received request for body type classification.")

    # Extract input parameters
    gender = data.get('gender')
    image_url = data.get('image_url')

    # Validate input data
    if not image_url or gender not in [0, 1, "0", "1"]:
        model_logger.warning("Invalid input data - missing image URL or invalid gender value.")
        return jsonify({
            "ok": False,
            "data": None,
            "error": {
                "code": "INVALID_INPUT",
                "message": "Invalid input data - missing image URL or invalid gender value"
            }
        }), 400

    # Download image
    img_path = download_image(image_url)
    if not img_path:
        model_logger.error("Failed to download image from provided URL.")
        return jsonify({
            "ok": False,
            "data": None,
            "error": {
                "code": "IMAGE_DOWNLOAD_FAILED",
                "message": "Failed to download image from provided URL"
            }
        }), 500

    # Process body type based on gender
    if str(gender) == "1":  # Male
        model_logger.info("Processing body type classification for male.")
        process_data = get_man_body_type(img_path)
    else:  # Female
        model_logger.info("Processing body type classification for female.")
        process_data = get_body_type_female(img_path)

    model_logger.info("Body type classification completed successfully.")
    return jsonify(process_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
