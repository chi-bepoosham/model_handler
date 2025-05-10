# Model Handler Service

This service processes clothing images and determines body types using machine learning models. It integrates with RabbitMQ for message queuing and handles both male and female clothing analysis.

## Features

- Gender-specific body type analysis
- Clothing image processing
- Asynchronous message processing using RabbitMQ
- Support for both male and female models
- Temporary image storage and processing

## Project Structure

```
/
├── .env                    # Environment variables (not in git)
├── .env.example           # Example environment variables
├── pyproject.toml         # Project configuration
├── setup.py               # Minimal setup file for editable installs
├── README.md              # Documentation
├── main.py                # Entry point script
└── src/
    └── model_handler_service/
        ├── __init__.py                   # Package initialization
        ├── config.py                     # Configuration management
        ├── load_and_predict_man.py       # Man related functions and predictors
        ├── load_and_predict_woman.py     # Woman related functions and predictors
        └── core/
            ├── __init__.py    # Core module initialization
            ├── messaging.py   # RabbitMQ communication
            ├── processing.py  # Image processing utilities
            └── utils.py       # General utilities
```

## Prerequisites

- Python 3.8 or higher
- RabbitMQ Server
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
# RabbitMQ Configuration
RABBITMQ_USER=develop
RABBITMQ_PASSWORD=your_password_here
RABBITMQ_HOST=rabbitmq
RABBITMQ_VHOST=rabbitmq
RABBITMQ_RESPONSE_QUEUE=ai_predict_process

# Image Processing Configuration
TEMP_IMAGES_DIR=/var/www/temp_images/

# Service Configuration
DEBUG=False
LOG_LEVEL=INFO
```

## Running the Service

### Running Locally
```bash
python main.py
```

### Running with Docker
```bash
docker build -t model-handler-service .
docker run -d model-handler-service
```

## API Flow

1. The service listens to the RabbitMQ queue for incoming messages
2. When a message is received, it processes the following data:
   - User ID
   - Gender
   - Clothes ID
   - Image link
   - Action type (body_type or clothing analysis)
3. The results are sent back to the response queue

## Response Formats

### Body Type Prediction Response

#### Success Response
```json
{
    "ok": true,
    "data": {
        "body_type": "string"  // One of the following values:
                               // For men: 'men_inverted_triangle', 'men_oval', 'men_rectangle'
                               // For women: "11", "21", "31", "41", "51"
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

### Body Type Values

#### Male Body Types
- "0": Ectomorph (Slim)
- "2": Mesomorph (Athletic)
- "5": Endomorph (Stocky)

#### Female Body Types
- "11": Hourglass
- "21": Pear
- "31": Apple
- "41": Rectangle
- "51": Athletic

### Woman's Clothing Image Processing Response

#### Success Response
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
        "skirt_type": "string"            // One of: "alineskirt", "balloonskirt", "mermaidskirt", 
                                        // "miniskirt", "pencilskirt", "shortaskirt", "wrapskirt"
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

#### Error Response - Image Download Failed



### Six Model Predictions Response

When the initial prediction fails, the service falls back to six model predictions:

```json
{
    "balted": "string",      // "balted" or "notbalted"
    "cowl": "string",        // "cowl" or "notcowl"
    "empire": "string",      // "empire" or "notempire"
    "loose": "string",       // "losse" or "snatched"
    "wrap": "string",        // "notwrap" or "wrap"
    "peplum": "string"       // "notpeplum" or "peplum"
}
```

## Development

### Project Architecture

The service is structured following a modular approach:

- **config.py**: Loads and validates environment variables
- **core/messaging.py**: Handles RabbitMQ communication
- **core/processing.py**: Contains image processing utilities
- **core/utils.py**: Provides common utility functions

### Adding New Models

To add new models, place them in the appropriate directory and update the main processing functions.
