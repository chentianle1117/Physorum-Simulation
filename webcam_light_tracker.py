import cv2
import numpy as np


class WebcamLightTracker:
    def __init__(self, debug_mode=True, mirror_horizontally=True, mirror_vertically=False):
        # Initialize webcam capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")

        # Debug mode flag
        self.debug_mode = debug_mode

        # Mirroring flags
        self.mirror_horizontally = mirror_horizontally
        self.mirror_vertically = mirror_vertically

    def process_frame(self):
        # Capture a frame from the webcam
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to grab frame")

        # Apply mirroring if enabled
        if self.mirror_horizontally:
            frame = cv2.flip(frame, 1)  # Flip horizontally (left-right)
        if self.mirror_vertically:
            frame = cv2.flip(frame, 0)  # Flip vertically (up-down)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to detect bright spots (light source)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours of the light spots
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If there are contours, find the largest one (assuming it's the light source)
        light_pos = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)

            # Compute the center of the largest contour
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                light_pos = (cx, cy)

                # Draw a circle around the detected light position
                if self.debug_mode:
                    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)  # Green contour
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red dot at light position

        # If in debug mode, show the frame with detected light position
        if self.debug_mode:
            cv2.imshow("Light Detection Debug", frame)

        return frame, light_pos

    def release(self):
        # Release the webcam when done
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        """Run the webcam and display the debug window in a loop."""
        while True:
            # Process the frame from the webcam
            _, light_pos = self.process_frame()

            # Wait for a key press and break the loop if ESC is pressed
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                print("Exiting webcam light tracker...")
                break

        # Release resources after exiting the loop
        self.release()


if __name__ == '__main__':
    # Create an instance of the WebcamLightTracker with debugging enabled
    light_tracker = WebcamLightTracker(debug_mode=True)

    # Run the light tracker, which will open a window to show the webcam feed and debug info
    light_tracker.run()
