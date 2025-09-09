import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

class WasteClassifier:
    def __init__(self):
        # Load pre-trained MobileNetV2 model
        self.model = MobileNetV2(weights='imagenet')
        
        # Define waste categories mapping
        self.waste_categories = {
            'plastic': ['bottle', 'plastic', 'container'],
            'paper': ['paper', 'cardboard', 'book', 'notebook'],
            'metal': ['can', 'tin', 'aluminum'],
            'organic': ['food', 'fruit', 'vegetable']
        }
    
    def preprocess_image(self, img_path):
        """Prepare image for model input"""
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x
    
    def classify_waste(self, img_path):
        """Classify waste from image"""
        # Preprocess the image
        processed_img = self.preprocess_image(img_path)
        
        # Get predictions
        predictions = self.model.predict(processed_img)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        # Map predictions to waste categories
        detected_categories = []
        for _, label, confidence in decoded_predictions:
            for category, keywords in self.waste_categories.items():
                if any(keyword in label for keyword in keywords):
                    detected_categories.append((category, confidence))
        
        if detected_categories:
            # Return the category with highest confidence
            category, confidence = max(detected_categories, key=lambda x: x[1])
            return category, confidence
        else:
            return "unknown", 0.0

if __name__ == "__main__":
    # Test the classifier
    classifier = WasteClassifier()
    image_path = "test_image.jpg"  # Replace with your test image
    category, confidence = classifier.classify_waste(image_path)
    print(f"Detected waste category: {category} (confidence: {confidence:.2f})")
