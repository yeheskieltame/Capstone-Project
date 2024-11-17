import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from datetime import datetime
import os


class WasteClassifier:
    def __init__(self, model_path='waste_classification_final_model.keras'):
        """
        Initialize the Waste Classifier
        Args:
            model_path: Path to the trained model file
        """
        print("Loading model...")
        self.model = load_model(model_path)
        self.class_names = ['Cardboard', 'Glass', 'Metal', 'Paper',
                            'Plastics', 'Textile Trash', 'Vegetation', 'organics']

        # Color palette for different classes (BGR format)
        self.colors = {
            'Cardboard': (139, 69, 19),  # Brown
            'Glass': (139, 69, 19),  # White
            'Metal': (192, 192, 192),  # Silver
            'Paper': (255, 235, 205),  # Light yellow
            'Plastics': (0, 255, 255),  # Yellow
            'Textile Trash': (255, 0, 255),  # Magenta
            'Vegetation': (0, 255, 0),  # Green
            'organics': (0, 128, 0)  # Dark green
        }

        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.recording = False
        self.out = None

        # ROI (Region of Interest) settings
        self.roi_active = False
        self.drawing = False
        self.roi_points = []

    def get_available_cameras(self):
        """Check available cameras"""
        available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras

    def preprocess_frame(self, frame):
        """Preprocess frame for prediction"""
        resized = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_frame / 255.0
        preprocessed = np.expand_dims(normalized, axis=0)
        return preprocessed

    def draw_detection_box(self, frame, prediction, confidence):
        """Draw detection box and label on frame"""
        frame_height, frame_width = frame.shape[:2]

        # Calculate center detection box (60% of frame size)
        box_width = int(frame_width * 0.6)
        box_height = int(frame_height * 0.6)
        x1 = (frame_width - box_width) // 2
        y1 = (frame_height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height

        # Get color for predicted class
        color = self.colors.get(prediction, (0, 255, 0))

        # Draw detection box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label = f"{prediction} ({confidence:.1f}%)"

        # Get label size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Draw label background
        cv2.rectangle(frame,
                      (x1, y1 - label_height - 10),
                      (x1 + label_width + 10, y1),
                      color,
                      -1)

        # Draw label text
        cv2.putText(frame,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

        return frame

    def draw_info_overlay(self, frame):
        """Draw information overlay"""
        # Draw FPS
        cv2.putText(frame,
                    f"FPS: {self.fps}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        # Draw recording indicator
        if self.recording:
            cv2.circle(frame,
                       (frame.shape[1] - 30, 30),
                       10,
                       (0, 0, 255),
                       -1)

        return frame

    def draw_detection_guide(self, frame):
        """Draw detection guide lines"""
        frame_height, frame_width = frame.shape[:2]

        # Draw center crosshair
        center_x = frame_width // 2
        center_y = frame_height // 2

        # Horizontal line
        cv2.line(frame,
                 (center_x - 20, center_y),
                 (center_x + 20, center_y),
                 (255, 255, 255),
                 1)

        # Vertical line
        cv2.line(frame,
                 (center_x, center_y - 20),
                 (center_x, center_y + 20),
                 (255, 255, 255),
                 1)

        return frame

    def start_recording(self, output_path='recordings'):
        """Start recording video"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_path, f'waste_classification_{timestamp}.avi')

        return filename

    def run(self, camera_index=0, display_scale=1.0):
        """Run real-time classification"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Cannot open camera {camera_index}")

        # Get frame size
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate display dimensions
        display_width = int(frame_width * display_scale)
        display_height = int(frame_height * display_scale)

        print("\nControls:")
        print("Press 'q' to quit")
        print("Press 'r' to start/stop recording")
        print("Press 's' to save current frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Update FPS
            self.frame_count += 1
            if time.time() - self.start_time >= 1:
                self.fps = self.frame_count
                self.frame_count = 0
                self.start_time = time.time()

            # Make prediction
            preprocessed = self.preprocess_frame(frame)
            predictions = self.model.predict(preprocessed, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_index] * 100
            predicted_class = self.class_names[predicted_class_index]

            # Draw detection box and label
            frame = self.draw_detection_box(frame, predicted_class, confidence)

            # Draw additional overlays
            frame = self.draw_info_overlay(frame)
            frame = self.draw_detection_guide(frame)

            # Record if active
            if self.recording and self.out:
                self.out.write(frame)

            # Resize for display
            if display_scale != 1.0:
                display_frame = cv2.resize(frame, (display_width, display_height))
            else:
                display_frame = frame

            # Show frame
            cv2.imshow('Waste Classification', display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                if not self.recording:
                    filename = self.start_recording()
                    self.out = cv2.VideoWriter(
                        filename,
                        cv2.VideoWriter_fourcc(*'XVID'),
                        20.0,
                        (frame_width, frame_height)
                    )
                    self.recording = True
                    print(f"\nStarted recording: {filename}")
                else:
                    self.recording = False
                    self.out.release()
                    self.out = None
                    print("\nStopped recording")
            elif key == ord('s'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'snapshot_{timestamp}.jpg'
                cv2.imwrite(filename, frame)
                print(f"\nSaved snapshot: {filename}")

        # Cleanup
        if self.recording:
            self.out.release()
        cap.release()
        cv2.destroyAllWindows()


def main():
    try:
        # Initialize classifier
        classifier = WasteClassifier()

        # Check available cameras
        available_cameras = classifier.get_available_cameras()

        if not available_cameras:
            print("No cameras found!")
            return

        print("\nAvailable cameras:", available_cameras)

        # Let user select camera if multiple are available
        if len(available_cameras) > 1:
            camera_index = int(input(f"Select camera index {available_cameras}: "))
        else:
            camera_index = available_cameras[0]

        # Let user select display scale
        scale = float(input("Enter display scale (0.5-2.0, default 1.0): ") or 1.0)

        # Start classification
        classifier.run(camera_index=camera_index, display_scale=scale)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
