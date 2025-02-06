import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
import time

# 🟢 Load the CNN model (MobileNetV2)
model = MobileNetV2(weights="imagenet")

# 🔹 Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 🔹 Email Credentials (Replace with your details)
EMAIL_SENDER = "karthicksaran2605@gmail.com"
EMAIL_RECEIVER = "karthicksaran2605sjce@gmail.com"
EMAIL_PASSWORD = "oxni zdmg rynu avxl"  # Your App Password

# 🔹 Function to send email
def send_email(human_images):
    try:
        message = MIMEMultipart()
        message["From"] = EMAIL_SENDER
        message["To"] = EMAIL_RECEIVER
        message["Subject"] = "Security Alert: Humans Detected!"

        message_body = f"ALERT - {len(human_images)} human(s) detected!"
        message.attach(MIMEText(message_body, "plain"))

        for i, img in enumerate(human_images):
            _, buffer = cv2.imencode(".jpg", img)
            image_data = buffer.tobytes()
            image_attachment = MIMEImage(image_data, name=f"detected_human_{i+1}.jpg")
            message.attach(image_attachment)

        # 🔹 Send email via SMTP
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, message.as_string())
        server.quit()

        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# 🔹 Function to preprocess frame
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to 224x224 for MobileNetV2
    frame_array = image.img_to_array(frame_resized)  # Convert to array
    frame_array = np.expand_dims(frame_array, axis=0)  # Add batch dimension
    frame_array = preprocess_input(frame_array)  # Normalize
    return frame_array

# 🔹 Function to detect humans using Haarcascades + MobileNetV2
def detect_human(frame):
    # 🔸 Step 1: Check for Faces using OpenCV Haarcascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        print("👀 Face detected! Likely a human.")
        return True  # If a face is detected, return True immediately

    # 🔸 Step 2: Use MobileNetV2 for object classification
    frame_preprocessed = preprocess_frame(frame)
    predictions = model.predict(frame_preprocessed)
    top_pred_class = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]

    # 🔹 Human-related labels
    human_labels = ["person", "man", "woman", "boy", "girl", "runner", "worker", "baby", "athlete"]

    # 🔹 Set a lower threshold (0.4)
    if top_pred_class[1] in human_labels and top_pred_class[2] > 0.4:
        print(f"🔴 Human detected! Label: {top_pred_class[1]} | Confidence: {top_pred_class[2]:.2f}")
        return True

    return False  # No human detected

# 🔹 Function to start detection
def start_detection():
    cap = cv2.VideoCapture(0)  # Use webcam (0) or change to video file path

    last_email_time = 0
    cooldown_time = 10  # ⏳ Reduce cooldown for faster testing

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)  # Flip for better display

        # 🔹 Detect human using both Haarcascade + MobileNetV2
        if detect_human(frame):
            current_time = time.time()
            if (current_time - last_email_time) > cooldown_time:
                send_email([frame])  # Send email with detected human image
                last_email_time = current_time  # Reset cooldown timer

        # 🖥️ Show the camera feed
        cv2.imshow("Human Detection Camera", frame)

        # ⏹️ Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the detection
if __name__ == "__main__":
    start_detection()
