# Dice Recognizer
A Python dice recognizer script that uses OpenCV to detect and count the number
of pips (dots) on the faces of 6-sided dice in a video stream from a webcam.

This script recognizes 6 sided dice with pips, either a single die or multiple
dice, counts the number of pips on each dice face and displays the total number
rolled. It is built to account for variable lighting if the camera has the
ability to focus and applies some basic preprocessing on the frame by first
gray scale the image, then applie a Gaussian blur to improve the thresholding
and contours.

The bounding boxes are actually used to detect the dice frames and later the
bounding circles for visual representation of what was actually detected. This
script seems to work well with various dice of multiple colors and sizes
(including translucent dice that throw off weird reflections).

There is a second window for settings and trackbars to adjust the settings in
real time to aid with calibration in various lighting environments. I keep it
around 1.8 or 0.9 depending on the lighting. The default threshold for
circularity is 0.6.

Note: the detection happens with rectangular bounding boxes because it's more
accurate, the circular bounding boxes are drawn to show what was detected to
the user and should be on the edges of the pips. The program was tested with
dice that have white pips: however, it was also tested with white dominos that
have black pips and works as well.

## Usage

Run with Python or make the script executable, your choice:

```python
python3 dice-recognizer.py
```

While the script is running:

```
Press q to quit.
```

# Acknowlegdements

The development of this application benefited from the assistance of language
models, including GPT-3.5 and GPT-4, provided by OpenAI. The author
acknowledges the valuable contributions made by these language models in
generating design ideas and providing insights during the development process.
