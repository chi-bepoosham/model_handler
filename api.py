
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
    model_logger.debug(f"Attempting to download image from URL: {image_url}")
    try:
        model_logger.debug(f"Making GET request to {image_url} with timeout 10s.")
        response = requests.get(image_url, timeout=10)
        model_logger.debug(f"Received response with status code: {response.status_code}")
        response.raise_for_status()
        model_logger.debug("Response status code indicates success.")

        filename = image_url.split('/')[-1]
        model_logger.debug(f"Extracted filename from URL: {filename}")
        # Check if the filename has a common image extension
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        has_extension = any(filename.lower().endswith(ext) for ext in image_extensions)
        model_logger.debug(f"Filename has common extension: {has_extension}")

        # If no common extension is found, add a default one (e.g., .jpg)
        if not has_extension:
            filename += '.jpg'
            model_logger.debug(f"Added default .jpg extension. New filename: {filename}")

        # Basic filename sanitization to prevent path traversal or invalid characters
        safe_filename = "".join([c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')]).rstrip()
        model_logger.debug(f"Sanitized filename: {safe_filename}")
        if not safe_filename: # Handle cases where filename becomes empty after sanitization
             safe_filename = "downloaded_image.jpg" # Default filename if original is unusable
             model_logger.debug(f"Sanitized filename was empty, using default: {safe_filename}")

        img_data = response.content # Use content from the successful response
        img_name = os.path.join(TEMP_IMAGES_DIR, f"{int(time.time())}{random.randrange(100, 999)}-temp-{safe_filename}")
        model_logger.debug(f"Constructed temporary file path: {img_name}")

        model_logger.debug(f"Writing image data to file: {img_name}")
        with open(img_name, 'wb') as handler:
            handler.write(img_data)
        model_logger.debug("Image data successfully written to file.")

        model_logger.info(f"Successfully downloaded image from {image_url} to {img_name}")
        return img_name

    except requests.exceptions.RequestException as e:
        model_logger.error(f"Request error downloading image from {image_url}: {e}")
        raise

    except OSError as e:
        model_logger.error(f"File system error saving image from {image_url} to {TEMP_IMAGES_DIR}: {e}")
        raise

    except Exception as e:
        # Catch any other unexpected errors
        model_logger.error(f"An unexpected error occurred while downloading image from {image_url}: {e}")
        raise


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
    try:
        img_path = download_image(image_url)
    except Exception as e:
        # Catch any exception during download and return detailed error
        model_logger.error(f"Failed to download image from {image_url}: {e}", exc_info=True)
        return jsonify({
            "ok": False,
            "data": None,
            "error": {
                "code": "IMAGE_DOWNLOAD_FAILED",
                "message": "Failed to download image from provided URL",
                "detail": str(e) 
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
    try:
        img_path = download_image(image_url)
    except Exception as e:
        # Catch any exception during download and return detailed error
        model_logger.error(f"Failed to download image from {image_url}: {e}", exc_info=True)
        return jsonify({
            "ok": False,
            "data": None,
            "error": {
                "code": "IMAGE_DOWNLOAD_FAILED",
                "message": "Failed to download image from provided URL",
                "detail": str(e) 
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
    app.run(host='0.0.0.0', port=80)
