import cv2
import numpy as np
from waste_classifier import WasteClassifier

def test_live_classification():
    # Initialize camera and classifier
    cap = cv2.VideoCapture(0)
    classifier = WasteClassifier()
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Save current frame
        cv2.imwrite("temp_frame.jpg", frame)
        
        # Classify the frame
        category, confidence = classifier.classify_waste("temp_frame.jpg")
        
        # Display result on frame
        text = f"{category}: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Waste Classification', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting waste classification test...")
    print("Press 'q' to quit")
    test_live_classification()
