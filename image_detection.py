import cv2
import numpy as np
import pygame  # For playing sound
import imutils

# Initialize pygame mixer for playing sounds
pygame.mixer.init()
sound = pygame.mixer.Sound('soundfile.wav')  # Replace with your sound file name

# Load the sample image
try:
    sample_img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
    sample_filename = 'sample.jpg'
    if sample_img is None:
        print("Error: Sample image not found or failed to load.")
        exit()
except Exception as e:
    print(f"Error loading sample image: {e}")
    exit()

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ORB detector for feature matching
orb = cv2.ORB_create()

# Compute keypoints and descriptors for the sample image
kp1, des1 = orb.detectAndCompute(sample_img, None)
if des1 is None:
    print("Error: Could not detect keypoints in the sample image.")
    exit()

# Define FLANN based matcher
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Detection threshold for matches
DETECTION_THRESHOLD = 10

# Function to play the detection sound
def play_sound():
    sound.play()

while True:
    # Read the webcam frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture webcam frame.")
        break
    
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and descriptors in the frame
    kp2, des2 = orb.detectAndCompute(gray_frame, None)
    
    if des2 is not None:
        # Match descriptors between the sample image and the webcam frame
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Filter good matches using Lowe's ratio test
        good_matches = []
        for match in matches:
            # Ensure there are at least two matches to unpack
            if len(match) == 2:
                m, n = match
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        # Print the number of good matches found
        print(f"Good matches found: {len(good_matches)}")
        
        # If enough good matches are found, draw a box and display the filename
        if len(good_matches) > DETECTION_THRESHOLD:
            # Get the matching keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography and check if it's valid
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                # Only proceed if M is a valid matrix
                h, w = sample_img.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                
                # Display filename above the box
                x, y = np.int32(dst)[0][0]
                cv2.putText(frame, sample_filename, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Play sound
                play_sound()
            else:
                print("Homography matrix could not be computed.")
    
    # Show the webcam frame with detected box and text
    cv2.imshow('Webcam', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
