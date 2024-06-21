import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input image filename
image_path = "line.jpg"

# Load input image
input_image = cv2.imread(image_path)

# Check if the image is loaded correctly
if input_image is None:
    print("Error: Unable to load image")
else:
    print("Image dimensions:", input_image.shape)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Create a copy of the original image to draw lines on
    line_image = input_image.copy()

    # Draw detected lines on the copy of the original image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Create a subplot with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Display the original image
    axs[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Display the image with detected lines
    axs[1].imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Image with Detected Lines")
    axs[1].axis('off')

    # Show the plot
    plt.show()
