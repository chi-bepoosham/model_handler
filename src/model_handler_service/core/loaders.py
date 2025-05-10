import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
from ultralytics import YOLO
from torchvision import models
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.resnet import ResNet101, ResNet152 
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from model_handler_service.core.processing import prepare_image, load_and_preprocess_image_paintane, mnist_prepar


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, class_num, base_model):
    """
    Load and configure a neural network model based on specified architecture.
    
    Args:
        model_path (str): Path to model weights file
        class_num (int): Number of output classes
        base_model (str): Model architecture type
        
    Returns:
        Model: Configured Keras model with loaded weights
    """
    if base_model == "resnet101":
        reso = 300
        input_shape = (reso, reso, 3)
        
        input_tensor = tf.keras.layers.Input(shape=input_shape)
        base_model = ResNet101(weights=None, include_top=False, 
                             input_shape=input_shape, input_tensor=input_tensor)
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(300, activation='relu')(x)
        predictions = Dense(class_num, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(model_path)
        return model

    elif base_model == "mobilenet":
        input_shape = (224, 224, 1)
        base_model = MobileNetV2(weights='imagenet', include_top=False,
                                input_shape=(224, 224, 3))
        
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(3, (1, 1), activation='relu')(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(class_num, activation='softmax')(x)
        
        model = tf.keras.models.Model(inputs, outputs)
        if model is None:
            raise ValueError("Failed to create the model!")
            
        model.load_weights(model_path)
        return model

    elif base_model == "resnet152":
        reso = 300
        input_shape = (reso, reso, 3)
        
        input_tensor = tf.keras.layers.Input(shape=input_shape)
        base_model = ResNet152(weights=None, include_top=False,
                             input_shape=input_shape, input_tensor=input_tensor)
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(300, activation='relu')(x)
        x = Dense(30, activation="relu")(x)
        predictions = Dense(class_num, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(model_path)
        return model
        
    elif base_model == "resnet152_600":
        reso = 300
        input_shape = (reso, reso, 3)
        
        input_tensor = tf.keras.layers.Input(shape=input_shape)
        base_model = ResNet152(weights=None, include_top=False,
                             input_shape=input_shape, input_tensor=input_tensor)
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(600, activation='relu')(x)
        x = Dense(30, activation="relu")(x)
        predictions = Dense(class_num, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(model_path)
        return model
                
    elif base_model == "mobilenet-v2":
        base_model = MobileNetV2(include_top=False, weights=None,
                                input_shape=(224, 224, 3))
        
        # Freeze first 50 layers
        for layer in base_model.layers[:50]:
            layer.trainable = False
        for layer in base_model.layers[50:]:
            layer.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)
        output_layer = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=base_model.input, outputs=output_layer)
        model.load_weights(model_path)
        return model
    
    elif base_model=="mobilenet-v2-pt":
    
        base_model = MobileNetV2(
        include_top=False,
        weights=None,  # Using trained weights from model_path
        input_shape=(224, 224, 3)
        )
        
        base_model.trainable = False

        # Adding custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(0.3)(x)
        output_layer = Dense(3, activation="softmax")(x) 

        model = Model(inputs=base_model.input, outputs=output_layer)
        model.load_weights(model_path)
        
        return model
        
    elif base_model == "mobilenet-v2-softmax":
        base_model = MobileNetV2(include_top=False, weights=None,
                                input_shape=(300, 300, 3))
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.4)(x)
        output_layer = Dense(class_num, activation="softmax")(x)

        model = Model(inputs=base_model.input, outputs=output_layer)
        model.load_weights(model_path)
        return model

    elif base_model == "mnist":
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(BatchNormalization())

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(10, activation='softmax'))
        model.load_weights(model_path)
        return model
    
    elif base_model=="resnet101_30_unit":
        reso = 300
        input_shape = (reso, reso, 3)

        # 1. بازسازی معماری مدل ResNet101
        input_tensor = tf.keras.layers.Input(shape=input_shape)
        base_model = ResNet101(weights=None, include_top=False, input_shape=input_shape, input_tensor=input_tensor)

        # افزودن لایه‌های بالا به مدل
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(300, activation='relu')(x)  # لایه Dense
        x = Dense(30, activation='relu')(x)  # لایه Dense

        predictions = Dense(class_num, activation='softmax')(x)  # لایه خروجی برای 11 کلاس

        # ساخت مدل کامل
        model = Model(inputs=base_model.input, outputs=predictions)

        # 2. لود وزن‌های ذخیره‌شده
        model.load_weights("{0}".format(model_path))
        return model
    
    elif base_model == "yolo":
        model = YOLO(model_path)
        model.to("cpu")
        return model
    
    elif base_model == "bodytype":
        # Initialize the model
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Linear(model.classifier.in_features, 3)
        model = model.to(device)

        for param in model.features.denseblock1.parameters():
            param.requires_grad = True

        for param in model.features.denseblock2.parameters():
            param.requires_grad = True

        for param in model.features.denseblock3.parameters():
            param.requires_grad = True

        for param in model.features.denseblock4.parameters():
            param.requires_grad = True

        # Load the state dictionary
        model.load_state_dict(torch.load(model_path, map_location=device))
        # Set the model to evaluation mode
        model.eval()
        return model


def predict_class(img, model, class_names, reso, model_name=None):
    """
    Predicts the class of an image using a given model.

    Args:
        img: The input image (NumPy array or image path)
        model: The loaded Keras model for prediction
        class_names: List of class names corresponding to model's output classes
        reso: Resolution to resize input image
        model_name: Optional name of model for logging/debugging

    Returns:
        str: Predicted class label
    """
    # Prepare image
    if model_name == "paintane" and len(class_names) == 2:
        img_array = load_and_preprocess_image_paintane(img)
    elif model_name == "mnist":
        img_array = mnist_prepar(img)
    elif model_name == "bodytype":
        # Make prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
        
        # Convert prediction to body type label
        body_type_labels = class_names
        predicted_body_type = body_type_labels[predicted.item()]
        
        return predicted_body_type
    else:
        img_array = prepare_image(img, target_size=(reso, reso))

    # Get prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=-1)
    predicted_label = class_names[predicted_class_index[0]]

    return predicted_label


def yolo_predict_crop(model, image_path, image=None):
    """
    Predicts and crops sleeve and collar regions from an image using YOLOv8.
    
    Args:
        model: YOLOv8 model for detection
        image_path: Path to input image
        image: Optional pre-loaded image array
        
    Returns:
        tuple: Cropped sleeve and collar images (both can be None if not detected)
    """
    # Get image filename
    image_file = os.path.basename(image_path)
    
    # Run detection
    results = model.predict(source=image_path, save=False)
    if not results or not results[0].boxes:
        print(f"No detections found in {image_file}")
        return None, None

    # Load image if not provided
    if image is None:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_file}")
            return None, None

    # Initialize crop variables
    crop_image_astin = None  # Sleeve crop
    crop_image_yaghe = None  # Collar crop

    # Process detections
    for result in results[0].boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        
        # Get detection class
        label_name = model.names[int(result.cls[0])]
        
        # Crop detected region
        cropped_image = image[y1:y2, x1:x2]
        if cropped_image.size == 0:
            continue

        # Store first valid detection of each type
        if label_name == 'sleeve' and crop_image_astin is None:
            crop_image_astin = cropped_image
        elif label_name == 'collar' and crop_image_yaghe is None:
            crop_image_yaghe = cropped_image

    return crop_image_astin, crop_image_yaghe
