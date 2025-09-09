import cv2
import os
from cvzone.HandTrackingModule import HandDetector  # Import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if webcam opens
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()  # Exit if webcam cannot be opened
else:
    print("Webcam is working")

# Set camera resolution (increase these values for better quality)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

# Gesture dataset folder
dataset_folder = r"D:\NLP Mini Project\DATAPICS"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# List of letters
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space"]

# Create folders for each letter
for letter in letters:
    letter_folder = os.path.join(dataset_folder, letter)
    if not os.path.exists(letter_folder):
        os.makedirs(letter_folder)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Start collecting gestures
current_letter = None

while True:
    ret, frame = cap.read()  # Capture frame from webcam
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Flip the frame for a natural mirror effect
    frame = cv2.flip(frame, 1)

    # Detect hands
    hands, frame = detector.findHands(frame)

    # Show live webcam feed with detected hands
    cv2.imshow("Capture Gesture", frame)

    key = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, quit the program
    if key == ord('q'):
        break

    # If a letter key is pressed, set the current gesture letter
    if key in [ord(letter) for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]:
        current_letter = chr(key)
        print(f"Collecting data for: {current_letter}")
        continue

    # If space is pressed, set the current gesture letter to 'space'
    if key == ord(' '):
        current_letter = "space"
        print("Collecting data for: space")
        continue

    # If a hand is detected and 'enter' is pressed, save the current frame as an image
    if hands and key == 13 and current_letter is not None:  # Enter key is pressed
        # Get the bounding box of the detected hand
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Add a margin around the bounding box to include more of the hand
        margin = 20  # Increase or decrease this value to adjust the crop area
        x = max(x - margin, 0)  # Ensure x is not negative
        y = max(y - margin, 0)  # Ensure y is not negative
        w = w + 2 * margin  # Expand width
        h = h + 2 * margin  # Expand height

        # Crop the hand from the frame using the expanded bounding box
        hand_crop = frame[y:y+h, x:x+w]
        
        folder_path = os.path.join(dataset_folder, current_letter)
        image_count = len(os.listdir(folder_path))  # Count how many images are already there
        image_path = os.path.join(folder_path, f"{current_letter}_{image_count+1}.jpg")  # Save with unique number
        
        # Save the cropped hand image
        cv2.imwrite(image_path, hand_crop)
        print(f"Image saved: {image_path}")

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
