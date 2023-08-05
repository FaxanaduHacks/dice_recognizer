#!/usr/bin/env python3

import cv2
import numpy as np

# This class is responsible for recognizing and processing dice faces in a
# video stream and incapsulates the functionality to detect, recognize and
# track dice faces:
class DiceRecognizer:
    def __init__(self):
        self.bounding_boxes = []    # List to store dice bounding boxes.
        self.threshold_value = 0    # Initial threshold for image binarization.
        self.threshold_max = 255    # Max threshold value.
        self.aspect_ratio_min = 0.9 # Min aspect ratio of bounding boxes.
        self.aspect_ratio_max = 1.2 # Max aspect ratio of bounding boxes.
        self.dice_values = []       # List to store recognized dice values.
        self.vote_duration = 25     # Number of frames to consider for voting.
        self.vote_threshold = 0.9   # Min ratio of votes required for value.

        # Create a setting window to hold the trackbars:
        self.create_trackbars_window()

    # Callback functions for the trackbars:
    def on_threshold_change(self, value):
        self.threshold_value = value

    def on_aspect_ratio_min_change(self, value):
        self.aspect_ratio_min = value / 10.0

    def on_aspect_ratio_max_change(self, value):
        self.aspect_ratio_max = value / 10.0

    # Create a window with trackbars to adjust some parameters used in the dice
    # recognition algorithm:
    def create_trackbars_window(self):
        # Create a window with trackbars to adjust parameters:
        cv2.namedWindow("Settings")
        cv2.createTrackbar("Threshold",
                           "Settings",
                           self.threshold_value,
                           self.threshold_max,
                           self.on_threshold_change)
        cv2.createTrackbar("Aspect Ratio Min",
                           "Settings",
                           int(self.aspect_ratio_min * 10),
                           20,
                           self.on_aspect_ratio_min_change)
        cv2.createTrackbar("Aspect Ratio Max",
                           "Settings",
                           int(self.aspect_ratio_max * 10),
                           20,
                           self.on_aspect_ratio_max_change)

    # Process the the input frames to detect and extract dice faces by
    # gray scaling the imagine, applying a Gaussian blur to reduce noise. It
    # uses imagine binarization using Otsu's thresholding method.
    def process_dice_frames(self, frame):
        # Preprocess the frame to detect and extract dice faces:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresholded = cv2.threshold(blur,
                                       self.threshold_value,
                                       self.threshold_max,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresholded.copy(),
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        dice_faces = []             # List to store dice face images.
        dice_bounding_boxes = []    # List to store dice bounding boxes.

        # Iterate over each contour detected in the image and for each contour
        # check if the contour area is greater than 700 (to exclude small or
        # noise-like contours that are unlikely to correspond to dice):
        for contour in contours:
            if cv2.contourArea(contour) > 700:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)

                # Filter contours based on aspect ratio:
                if self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max:
                    dice_face = gray[y:y + h, x:x + w]
                    dice_faces.append(dice_face)
                    dice_bounding_boxes.append((x, y, w, h))

                    # Calculate the center coordinates and radius of the
                    # circular bounding box for the detected dice face:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    radius = min(w, h) // 2

                    # Draw a circular bounding box on the frame around the
                    # detected pip (Cyan BGR:255,255,0):
                    cv2.circle(frame,
                               (center_x, center_y),
                               radius,
                               (255, 255, 0),
                               2)

        return dice_faces, dice_bounding_boxes

    # This method is responsible for recognizing the number of pips (dots) on a
    # single dice face:
    def recognize_dice_value(self, dice_face):
        # Recognize the value of a dice face by counting the circular pips:
        _, thresholded = cv2.threshold(dice_face,
                                       0,
                                       255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresholded.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Keep track of the number of pips detected:
        circular_pips = 0

        # Iterate through each contour and calculate the circularity of the
        # contour based on its area and perimeter. The default threshold value
        # for circularity is 0.6:
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                # Filter contours based on circularity:
                if circularity > 0.6:
                    circular_pips += 1

        return max(circular_pips - 1, 0)  # Ensure a minimum value of 0.

    # This method is responsible for updating and tracking the values obtained
    # from recognizing the pips on the dice faces:
    def update_dice_values(self, dice_value):
        # Update the list of recognized dice values and perform voting to
        # determine the most frequent value:
        self.dice_values.append(dice_value)

        # Keep only the most recent votes within the vote duration:
        self.dice_values = self.dice_values[-self.vote_duration:]

        # Perform voting:
        votes = np.bincount(self.dice_values)
        most_frequent_value = np.argmax(votes)

        # Calculate the ratio of the most frequent value to the total number
        # of votes:
        vote_ratio = votes[most_frequent_value] / len(self.dice_values)

        # If the vote ratio exceeds the threshold, return the recognized value:
        if vote_ratio >= self.vote_threshold:
            return most_frequent_value + 1
        else:
            return None # Otherwise, return None.

# Check if the current script is being run as the main module; if so, run it:
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)                   # Open the webcam.
    dice_recognizer = DiceRecognizer()          # Instansiate the class.
    dice_recognizer.create_trackbars_window()   # Create the trackbars window.

    # Number of consecutive empty frames to consider for debouncing:
    empty_frame_threshold = 10

    while True:
        ret, frame = cap.read() # Read a frame from the video stream.
        if not ret:             # If the frame reading fails, exit and loop.
            break

        # Process the frame to detect the dice:
        dice_faces, dice_bounding_boxes = dice_recognizer.process_dice_frames(frame)

        if dice_faces is None or dice_bounding_boxes is None:
            empty_frame_counter += 1
            if empty_frame_counter >= empty_frame_threshold:
                total_dice_value = 0
            continue
        else:
            empty_frame_count = 0

        # Update and display the total dice value on the frame:
        if dice_faces is not None and dice_bounding_boxes is not None:
            total_dice_value = 0
            for dice_face in dice_faces:
                # Recognize the value of each dice face:
                dice_value = dice_recognizer.recognize_dice_value(dice_face)

                # Update the recognized value:
                recognized_value = dice_recognizer.update_dice_values(dice_value)

                # If recognized_value is not None it means that a valid dice
                # value was recognized for a one of the dice faces and that
                # value can be added to total_dice_value:
                if recognized_value is not None:
                    total_dice_value += recognized_value

            # Display the total dice value on the frame (Cyan BGR:255,255,0):
            cv2.putText(frame, f"Total Dice Value: {total_dice_value}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2)

        # Display the processed frame with dice detection:
        cv2.imshow("Dice Recognizer", frame)

        # Create a blank frame for the trackbars window:
        trackbars_frame = np.zeros((150, 400, 3), np.uint8)

        # Display the current threshold value:
        trackbars_frame = cv2.putText(trackbars_frame,
                                      f"Threshold: {dice_recognizer.threshold_value}",
                                      (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6,
                                      (255, 255, 0),
                                      1)
        # Display the current mininum aspect ratio:
        trackbars_frame = cv2.putText(trackbars_frame,
                                      f"Aspect Ratio Min: {dice_recognizer.aspect_ratio_min}",
                                      (10, 60),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6,
                                      (255, 255, 0),
                                      1)

        # Display the current maximum aspect ratio:
        trackbars_frame = cv2.putText(trackbars_frame,
                                      f"Aspect Ratio Max: {dice_recognizer.aspect_ratio_max}",
                                      (10, 90),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6,
                                      (255, 255, 0),
                                      1)

        # Display the trackbars window:
        cv2.imshow("Settings", trackbars_frame)

        # Press q to quit:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()           # Release the video capture device.
    cv2.destroyAllWindows() # Close all windows.
