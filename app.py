import streamlit as st
import cv2
import numpy as np
from openvino.runtime import Core

# Load OpenVINO models
ie = Core()
face_detection_model = ie.read_model(model="models/face-detection-adas-0001.xml")
face_detection_compiled = ie.compile_model(face_detection_model, device_name="CPU")

emotions_model = ie.read_model(model="models/emotions-recognition-retail-0003.xml")
emotions_compiled = ie.compile_model(emotions_model, device_name="CPU")

face_reid_model = ie.read_model(model="models/face-reidentification-retail-0095.xml")
face_reid_compiled = ie.compile_model(face_reid_model, device_name="CPU")

# Helper function for face detection
def detect_faces(image):
    height, width = image.shape[:2]
    input_blob = cv2.resize(image, (672, 384))
    input_blob = np.transpose(input_blob, (2, 0, 1))
    input_blob = input_blob[np.newaxis, :]
    
    # Run inference on face detection model
    results = face_detection_compiled([input_blob])
    detections = results[face_detection_compiled.output(0)]
    
    boxes = []
    for detection in detections[0][0]:
        confidence = detection[2]
        if confidence > 0.5:
            xmin, ymin, xmax, ymax = (detection[3:7] * [width, height, width, height]).astype(int)
            boxes.append((xmin, ymin, xmax, ymax))
    
    return boxes

# Helper function for emotion recognition
def recognize_emotions(image, face_box):
    xmin, ymin, xmax, ymax = face_box
    face = image[ymin:ymax, xmin:xmax]
    face_resized = cv2.resize(face, (64, 64))
    face_blob = np.transpose(face_resized, (2, 0, 1))
    face_blob = face_blob[np.newaxis, :]
    
    # Run inference on emotion recognition model
    emotion_results = emotions_compiled([face_blob])
    emotions = emotion_results[emotions_compiled.output(0)][0]
    emotion_index = np.argmax(emotions)
    emotion_map = ["Neutral", "Happy", "Sad", "Surprised", "Angry"]
    return emotion_map[emotion_index]

# Helper function for face re-identification
def compare_faces(input_face, known_face):
    input_blob = cv2.resize(input_face, (128, 128))
    input_blob = np.transpose(input_blob, (2, 0, 1))
    input_blob = input_blob[np.newaxis, :]
    
    known_blob = cv2.resize(known_face, (128, 128))
    known_blob = np.transpose(known_blob, (2, 0, 1))
    known_blob = known_blob[np.newaxis, :]
    
    # Run inference on both images
    input_embedding = face_reid_compiled([input_blob])[face_reid_compiled.output(0)]
    known_embedding = face_reid_compiled([known_blob])[face_reid_compiled.output(0)]
    
    # Normalize embeddings using cv2.normalize
    input_embedding = cv2.normalize(input_embedding, None, alpha=0, beta=1, norm_type=cv2.NORM_L2)
    known_embedding = cv2.normalize(known_embedding, None, alpha=0, beta=1, norm_type=cv2.NORM_L2)
    
    # Compute Euclidean distance between the embeddings
    distance = np.linalg.norm(input_embedding - known_embedding)
    return distance

# Main app
st.title("Takoyaki Kiosk")

# Tabs for Login and Feedback
tab1, tab2 = st.tabs(["Login", "Feedback"])

# Login Tab
with tab1:
    st.header("Login")
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png"])
    
    if uploaded_file:
        # Read and decode the uploaded image
        img = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(img, 1)
        
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
        
        # Detect faces in the uploaded image
        boxes = detect_faces(img)
        
        if boxes:
            # Draw bounding boxes around the detected face
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            img_rgb_boxed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb_boxed, caption="Detected Face", use_column_width=True)
            
            # Load the known face image from the library
            known_face = cv2.imread("known_faces/known_faces.png")
            known_face_rgb = cv2.cvtColor(known_face, cv2.COLOR_BGR2RGB)
            
            # Extract the face region from the uploaded image
            face_detected = img[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2]]
            
            # Ensure the face size is appropriate
            st.write(f"Detected face size: {face_detected.shape}")
            
            # Compare the detected face with the known face
            distance = compare_faces(face_detected, known_face_rgb)
            
            # Debug: print distance
            st.write(f"Distance: {distance:.4f}")
            
            # Set a stricter threshold to only recognize the registered user
            if distance < 0.25:
                st.success("Face recognised! Welcome, User!")
            else:
                st.error(f"Face not recognised! (Distance: {distance:.4f})")

# Feedback Tab
with tab2:
    st.header("Feedback")
    feedback = st.selectbox("How was your experience today?", ["Good", "Poor", "Excellent"])
    
    if st.button("Submit Feedback"):
        # Start camera and take picture
        picture = st.camera_input("Take a picture")
        
        if picture:
            img = np.array(bytearray(picture.read()), dtype=np.uint8)
            img = cv2.imdecode(img, 1)
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces in the captured photo
            boxes = detect_faces(img)
            
            if boxes:
                # Draw bounding boxes around the detected face
                for box in boxes:
                    xmin, ymin, xmax, ymax = box
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                img_rgb_boxed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb_boxed, caption="Detected Face", use_column_width=True)
                
                # Recognize the emotion of the detected face
                detected_emotion = recognize_emotions(img, boxes[0])
                st.write("Thank you for your feedback!")
                st.write(f"Emotion detected: {detected_emotion}")
