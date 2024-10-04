import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import time  # Import the time module

# Use the generated App Password
password =   # App Password from Google
from_email =   # Your Gmail address
to_email =   # Receiver's email address

# Function to send email with images of detected humans
def send_email(to_email, from_email, human_images):
    try:
        # Create message
        message = MIMEMultipart()
        message["From"] = from_email
        message["To"] = to_email
        message["Subject"] = "Security Alert: Humans Detected!"

        message_body = f"ALERT - {len(human_images)} human(s) have been detected by the camera!"
        message.attach(MIMEText(message_body, "plain"))

        # Attach images of detected humans
        for i, image in enumerate(human_images):
            _, buffer = cv2.imencode('.jpg', image)  # Encode the image to jpg format
            image_data = buffer.tobytes()  # Convert the image to bytes
            image_attachment = MIMEImage(image_data, name=f"detected_human_{i+1}.jpg")
            message.attach(image_attachment)

        # Setup the SMTP server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        # Login to the SMTP server with the App Password
        server.login(from_email, password)

        # Send the email
        server.sendmail(from_email, to_email, message.as_string())
        server.quit()
        print("Email sent successfully!")

    except Exception as e:
        print(f"Failed to send email: {e}")

# Object detection class using YOLO for human detection
class HumanDetection:
    def __init__(self, capture_index=0, cooldown_time=60):
        """
        Initializes a HumanDetection instance with a given camera index and cooldown time for sending emails.
        """
        self.capture_index = capture_index
        self.last_email_time = 0
        self.cooldown_time = cooldown_time  # Cooldown time in seconds
        self.model = YOLO("yolov8n.pt")  # Load YOLO model (you can use other versions)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.annotator = None

    def __call__(self):
        """
        Method to start video capture and detect humans.
        """
        # Start video capture
        cap = cv2.VideoCapture(self.capture_index)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            # Flip the frame horizontally to correct the display
            frame = cv2.flip(frame, 1)

            # Run inference
            results = self.model(frame)

            # Annotate detected objects
            self.annotator = Annotator(frame)
            human_images = []  # List to hold images of detected humans
            human_count = 0

            # Iterate through each result in the list
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = result.names[int(box.cls.item())]

                    # Check if the detected object is a human (YOLO label for humans is 'person')
                    if label == "person":
                        human_count += 1
                        human_images.append(frame[y1:y2, x1:x2])  # Capture the image of the detected human
                        self.annotator.box_label([x1, y1, x2, y2], f"{label}: {box.conf.item():.2f}", color=colors(0))

            # Display annotated frame
            annotated_frame = self.annotator.result()
            cv2.imshow("YOLO Human Detection", annotated_frame)

            # Send email alert if humans are detected and cooldown period has passed
            current_time = time.time()  # Get current time
            if human_count > 0 and (current_time - self.last_email_time) > self.cooldown_time:
                send_email(to_email, from_email, human_images)  # Send images of detected humans
                self.last_email_time = current_time  # Reset email timer

            # Exit if 'ESC' key is pressed
            if cv2.waitKey(1) & 0xFF == 27:  # Check if ESC key is pressed
                print("Exiting...")
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# Run the human detection
if __name__ == "__main__":
    detector = HumanDetection(capture_index=0, cooldown_time=60)  # Cooldown set to 60 seconds
    detector()
