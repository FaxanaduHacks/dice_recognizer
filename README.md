# dice_recognizer

A Python program that utilizes computer vision techniques to detect and recognize dice faces from a video stream. It employs techniques such as image binarization, contour detection, circularity analysis, and voting to accurately identify the number of pips (dots) on each dice face. The program also provides an interactive settings window with trackbars to fine-tune parameters for better recognition performance.

This script recognizes 6-sided dice with pips, either a single die or multiple dice, counts the number of pips on each dice face and displays the total number rolled on the fly. It is built to account for variable lighting if the camera has the ability to focus and applies some basic preprocessing on the frame by first gray scaling the image, then applying a Gaussian blur to improve the thresholding
and contours.

The bounding boxes are actually used to detect the dice frames and later the bounding circles for visual representation of what was actually detected. This script seems to work well with various dice of multiple colors and sizes (including translucent dice that can throw off weird reflections).

There is a second window for settings and trackbars to adjust the settings in real-time to aid with calibration in various lighting environments. I recommend keeping it around 0.9 (aspect_ratio_min) and 1.2 (aspect_ratio_max), but this depends on the lighting. The default threshold for circularity is 0.6.

Note: The detection happens with rectangular bounding boxes because it's more accurate while the circular bounding boxes are drawn to show what was detected to the user and should be on the edges of the pips. The program was tested with dice that have white pips: however, it was also tested with white dominos that have black pips and works as well.

## Installation

1. Make sure you have Python 3 installed on your system.
2. Install the required packages using the following command:
   ```
   pip install opencv-python numpy
   ```

## Usage

1. Run the `dice_recognizer.py` script using the command:
   ```
   python dice_recognizer.py
   ```

2. A video stream from your default webcam will open, and the program will process the frames to detect and recognize dice faces.

3. Adjust the parameters using the trackbars window to achieve better recognition results.

4. Press the 'q' key to exit the program.

### Demo
![Camera Rig Setup](https://github.com/FaxanaduHacks/dice_recognizer/Backdrop.png "Using a white piece of paper for a backdrop, camera positioned facing the desk.")

![Dice Recognizer Demonstration](https://github.com/FaxanaduHacks/dice_recognizer/Demonstration.png "Demonstrates live detection and counting of pips, also shows the sliders window.")

## Features

- **Dice Detection:** The program detects dice faces in real-time using image preprocessing, binarization, and contour analysis.

- **Dice Recognition:** The number of pips (dots) on each dice face is recognized through circularity analysis of contours.

- **Settings Window:** An interactive settings window with trackbars allows you to adjust parameters such as threshold, aspect ratio min, and aspect ratio max.

- **Voting Mechanism:** A voting mechanism helps ensure accurate recognition results by considering multiple frames.

## Parameters

- `threshold_value`: Initial threshold for image binarization.
- `aspect_ratio_min`: Minimum aspect ratio of bounding boxes.
- `aspect_ratio_max`: Maximum aspect ratio of bounding boxes.
- `vote_duration`: Number of frames to consider for voting.
- `vote_threshold`: Minimum ratio of votes required for recognizing a value.

## Customization

Feel free to modify the program according to your requirements. You can experiment with parameter values, add more features, or integrate the dice recognition functionality into larger projects.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This program utilizes the OpenCV library for computer vision tasks. Credits to the OpenCV community for their contributions.

The development of this application benefited from the assistance of language models, including GPT-3.5 and GPT-4, provided by OpenAI. The author acknowledges the valuable contributions made by these language models in generating design ideas and providing insights during the development process.
