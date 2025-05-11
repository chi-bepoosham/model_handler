# Model Handler Service

This service processes clothing images and determines body types using machine learning models. It provides a REST API for clothing classification and body type prediction, supporting both male and female analysis.

## Features

- RESTful API for clothing and body type analysis
- Gender-specific body type analysis
- Clothing image processing and classification
- Color tone extraction
- Support for both male and female models
- Temporary image storage and processing

## Project Structure

```
/
├── .env                    # Environment variables (not in git)
├── .env.example            # Example environment variables
├── api.py                  # Flask API implementation
├── Dockerfile              # Docker configuration
├── monitor_memory.py       # Memory usage monitoring utility
├── pyproject.toml          # Project configuration
├── README.md               # Documentation
└── src/
    └── model_handler_service/
        ├── __init__.py              # Package initialization
        ├── color.py                 # Color extraction utilities
        ├── load_and_predict_man.py  # Male model prediction functionality
        ├── load_and_predict_woman.py# Female model prediction functionality
        ├── test.py                  # Testing utilities
        └── core/
            ├── __init__.py        # Core module initialization
            ├── config.py          # Configuration management
            ├── loaders.py         # Model loading utilities
            ├── logger.py          # Logging configuration
            ├── processing.py      # Image processing utilities
            └── validations.py     # Input validation utilities
```

## Prerequisites

- Python 3.8 or higher
- Flask
- Docker (for containerized deployment)

## Installation

### 1. Clone the repository:
```bash
git clone [repository-url]
cd model_handler_service
```

### 2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration values
```

### 3. Install the package:
```bash
# Install the base package
pip install -e .
```

## Configuration

The service requires the following environment variables in your `.env` file:

```
# Service Configuration
DEBUG=False
LOG_LEVEL=INFO

# Image Processing Configuration
TEMP_IMAGES_DIR=/path/to/temp/images/
```

## Running the Service

### Running Locally
```bash
python api.py
```

### Running with Docker
```bash
docker build -t model-handler-service .
docker run -d -p 5001:5001 model-handler-service
```

## API Endpoints

The service exposes two main endpoints:

### 1. `/clothing` - Clothing Classification

**Method**: POST

**Request Body**:
```json
{
    "gender": "0",  // "0" for female, "1" for male
    "image_url": "https://example.com/image.jpg"
}
```

**Response**:
See the Response Formats section below.

### 2. `/bodytype` - Body Type Classification

**Method**: POST

**Request Body**:
```json
{
    "gender": "0",  // "0" for female, "1" for male
    "image_url": "https://example.com/image.jpg"
}
```

**Response**:
See the Response Formats section below.

## Response Formats

### Body Type Prediction Response

#### Success Response
```json
{
    "ok": true,
    "data": {
        "body_type": "string"  // One of the following values:
                               // For men: "0" (Ectomorph/Slim), "2" (Mesomorph/Athletic), "5" (Endomorph/Stocky)
                               // For women: "11" (Hourglass), "21" (Pear), "31" (Apple), "41" (Rectangle), "51" (Athletic)
    },
    "error": null
}
```

#### Error Response - Image Load Error
```json
{
    "ok": false,
    "data": null,
    "error": {
        "code": "IMAGE_LOAD_ERROR",
        "message": "Failed to load image file"
    }
}
```

#### Error Response - Human Validation Error
```json
{
    "ok": false,
    "data": null,
    "error": {
        "code": "HUMAN_VALIDATION_ERROR",
        "message": "Failed human body validation",
        "human_validation_errors": [
            // Array of validation error messages
        ]
    }
}
```

#### Error Response - Image Download Failed
```json
{
    "ok": false,
    "data": null,
    "error": {
        "code": "IMAGE_DOWNLOAD_FAILED",
        "message": "Failed to download image from provided URL"
    }
}
```

### Clothing Image Processing Response

#### Success Response for Women's Clothing
```json
{
    "ok": true,
    "data": {
        "color_tone": "string",           // Dominant color tone of the clothing
        "mnist_prediction": "string",     // "Under" or "Over" clothing
        "paintane": "string",             // One of: "fbalatane", "fpaintane", "ftamamtane"
        
        // Upper body predictions (if paintane is "fbalatane" or "ftamamtane")
        "astin": "string",                // One of: "bottompuffy", "fhalfsleeve", "flongsleeve", 
                                        // "fshortsleeve", "fsleeveless", "toppuffy"
        "pattern": "string",              // One of: "dorosht", "rahrahamudi", "rahrahofoghi", 
                                        // "riz", "sade"
        "yaghe": "string",                // One of: "boatneck", "classic", "halter", "hoodie", 
                                        // "of_the_shoulder", "one_shoulder", "round", "squer", 
                                        // "sweatheart", "turtleneck", "v_neck"
        
        // Lower body predictions (if paintane is "fpaintane" or "ftamamtane")
        "rise": "string",                 // "highrise" or "lowrise"
        "shalvar": "string",              // One of: "wbaggy", "wbootcut", "wcargo", "wcargoshorts", 
                                        // "wmom", "wshorts", "wskinny", "wstraight"
        "tarh_shalvar": "string",         // One of: "wpamudi", "wpdorosht", "wpofoghi", 
                                        // "wpriz", "wpsade"
        "skirt_and_pants": "string",      // "pants" or "skirt"
        "skirt_print": "string",          // One of: "skirtamudi", "skirtdorosht", "skirtofoghi", 
                                        // "skirtriz", "skirtsade"
        "skirt_type": "string",           // One of: "alineskirt", "balloonskirt", "mermaidskirt", 
                                        // "miniskirt", "pencilskirt", "shortaskirt", "wrapskirt"
                                        
        // Six model predictions (for upper body clothing)
        "balted": "string",      // "balted" or "notbalted"
        "cowl": "string",        // "cowl" or "notcowl"
        "empire": "string",      // "empire" or "notempire"
        "loose": "string",       // "losse" or "snatched"
        "wrap": "string",        // "notwrap" or "wrap"
        "peplum": "string"       // "notpeplum" or "peplum"
    },
    "error": null
}
```

#### Error Response - Clothing Validation Error
```json
{
    "ok": false,
    "data": null,
    "error": {
        "code": "CLOTHING_VALIDATION_ERROR",
        "message": "Failed clothing validation checks",
        "validation_errors": [
            // Array of validation error messages
        ]
    }
}
```

## Development

### Project Architecture

The service is structured following a modular approach:

- **api.py**: Flask API endpoints for clothing and body type classification
- **color.py**: Color extraction and analysis utilities
- **load_and_predict_man.py**: Male-specific clothing and body type prediction
- **load_and_predict_woman.py**: Female-specific clothing and body type prediction
- **core/config.py**: Loads and validates environment variables
- **core/loaders.py**: Handles model loading functions
- **core/logger.py**: Configures logging for the application
- **core/processing.py**: Contains image processing utilities
- **core/validations.py**: Input validation functions

### Adding New Models

To add new models, place them in the appropriate directory and update the relevant prediction functions in the processing modules.
