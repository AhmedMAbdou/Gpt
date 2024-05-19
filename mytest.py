import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Function to load TensorFlow Lite model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# Function to load labels from file
def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]
    return labels

# Function to preprocess image data and resize it
def preprocess_frame(frame, target_size=(224, 224)):
    resized_frame = cv2.resize(frame, target_size)
    processed_frame = resized_frame.astype(np.float32) / 255.0
    return processed_frame

# Function to perform object detection
def detect_objects(interpreter, input_details, output_details, frame):
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(frame, axis=0))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# Create the Streamlit app
st.title('Object Detection App')

# Load the TensorFlow Lite model
model_path = "models/mobilenet_v1_1.0_224.tflite"
interpreter, input_details, output_details = load_model(model_path)

# Load labels from file
labels_path = "models/mobilenet_v1_1.0_224.txt"
labels = load_labels(labels_path)

# Add a placeholder for displaying the captured image
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Load the image from the buffer
    image = Image.open(img_file_buffer)

    # Convert the image to a numpy array
    frame = np.array(image)

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Perform object detection on the frame
    output = detect_objects(interpreter, input_details, output_details, processed_frame)

    # Extract detected object label and confidence score
    detected_label = labels[np.argmax(output)]
    confidence = np.max(output)

    # Draw the detected label and confidence score on the frame
    text = f"{detected_label}   {confidence:.2f}"
    frame = cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert frame from BGR to RGB (if necessary, depends on model requirements)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the annotated frame
    st.image(frame_rgb, caption="Processed Image", use_column_width=True)
