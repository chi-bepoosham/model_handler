import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tensorflow.keras.utils import img_to_array
from keras.applications.resnet import preprocess_input


def prepare_image(img, target_size):
    img = cv2.resize(img,target_size)
    img_array = img_to_array(img)  # تبدیل به آرایه
    img_array = np.expand_dims(img_array, axis=0)  # افزودن بعد اضافی برای Batch
    img_array = preprocess_input(img_array)  # پیش‌پردازش تصویر برای ResNet
    return img_array


def load_and_preprocess_image_paintane(image_path):
    img = cv2.resize(image_path, (224, 224))
    img_array = img_to_array(img)  # Convert to NumPy array
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array


def mnist_prepar(image):
    image = cv2.resize(image, (224, 224)) 
    if len(image.shape) == 2: 
        image = np.expand_dims(image, axis=-1)  
        image = np.repeat(image, 3, axis=-1)  
    image = np.expand_dims(image, axis=0)  
    return image.astype(np.float32) / 255.0  


def preprocess_image_for_bodytype(image_path):
    """
    Preprocess a single image for bodytype
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        torch.Tensor: Preprocessed image tensor
        
    Raises:
        FileNotFoundError: If the image file does not exist
        IOError: If there is an error loading the image
        Exception: For other unexpected errors during preprocessing
    """
    try:
        # Define transforms for inference (simpler than training transforms)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to fixed size
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load and transform the image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        image = Image.open(image_path)
        if image is None:
            raise IOError(f"Failed to load image: {image_path}")
            
        image = image.convert('RGB')
        image_tensor = transform(image)
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
        
    except (FileNotFoundError, IOError) as e:
        raise e
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")
