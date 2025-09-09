import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import pyttsx3

# Initialize model and hand detector
model = load_model(r"D:\NLP Mini Project\Model\keras_model12.h5")
detector = HandDetector(maxHands=1)
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space"]
sentence = ""

# Text-to-speech conversion
def convert_text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Button press logic (Enter key to finalize letter)
def button_pressed():
    return cv2.waitKey(1) & 0xFF == 13  # Enter key

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Add padding for better hand capture
        padding = 20
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = min(img.shape[1] - x, w + 2 * padding), min(img.shape[0] - y, h + 2 * padding)

        # Preprocess hand image
        imgCrop = img[y:y+h, x:x+w]
        imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
        imgGray = cv2.resize(imgGray, (200, 200))
        imgGray = imgGray / 255.0
        imgGray = np.expand_dims(imgGray, axis=(0, -1))

        # Predict the letter
        prediction = model.predict(imgGray)
        confidence = np.max(prediction)

        if confidence > 0.8:
            index = np.argmax(prediction)
            predicted_letter = labels[index]
        else:
            predicted_letter = "?"  # Low confidence

        # Display predicted letter
        cv2.putText(imgOutput, predicted_letter, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        # Append letter if button pressed
        if button_pressed():
            sentence += " " if predicted_letter == "space" else predicted_letter
            print("Current sentence:", sentence)

    # Show the image with letter prediction
    cv2.imshow("Image", imgOutput)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Convert the formed sentence to speech
convert_text_to_speech(sentence)
print(imgCrop.shape)  # Color: (Height, Width, 3), Grayscale: (Height, Width)


# Release resources
cap.release()
cv2.destroyAllWindows()
