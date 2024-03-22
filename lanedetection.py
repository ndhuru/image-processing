import cv2
import numpy as np
import math

# Function to create a mask for the region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)  # Create a black image of the same size as img
    match_mask_color = 255  # Color to fill the region of interest with
    cv2.fillPoly(mask, vertices, match_mask_color)  # Fill the region of interest with white
    masked_image = cv2.bitwise_and(img, mask)  # Bitwise AND operation to get the region of interest
    return masked_image

# Function to draw lines on an image
def draw_lines(img, lines, thickness=5, outer_color=[255, 0, 255]):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # Create an empty image for lines
    img = np.copy(img)  # Create a copy of the input image
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # Skip lines with zero denominator to avoid division by zero
            slope = (y2 - y1) / (x2 - x1)  # Calculate the slope of the line
            if math.fabs(slope) < 0.5:  # Skip lines with slopes less than 0.5
                continue
            # Draw lines with appropriate slope
            cv2.line(line_img, (x1, y1), (x2, y2), outer_color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)  # Combine original image and lines image
    return img

# Main pipeline function to process the image
def pipeline(image):
    height, width, _ = image.shape  # Get height and width of the image
    # Define vertices of the region of interest
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    # Get the region of interest
    cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    if lines is None:
        return image  # Return original image if no lines detected

    # Initialize lists to store coordinates of left and right lines
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    # Iterate through detected lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)  # Calculate the slope of the line
            if math.fabs(slope) < 0.5:
                continue  # Skip lines with slopes less than 0.5
            # Separate lines based on their slopes
            if slope <= 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    # Define variables for left and right lines and their intersection with horizon
    poly_left = poly_right = None
    left_x_start = left_x_end = right_x_start = right_x_end = 0
    left_x_horizon = right_x_horizon = 0

    # Fit a polynomial to left lines if any
    if left_line_x and left_line_y:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
        left_x_start = int(poly_left(image.shape[0]))
        left_x_end = int(poly_left(image.shape[0] * (3 / 5)))
        left_x_horizon = int(poly_left(image.shape[0]))  # Intersection with horizon
    # Fit a polynomial to right lines if any
    if right_line_x and right_line_y:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        right_x_start = int(poly_right(image.shape[0]))
        right_x_end = int(poly_right(image.shape[0] * (3 / 5)))
        right_x_horizon = int(poly_right(image.shape[0]))  # Intersection with horizon

    # Calculate the centerline based on left and right lines
    center_x_start = (left_x_start + right_x_start) // 2
    center_x_end = (left_x_end + right_x_end) // 2
    horizon_y = image.shape[0] // 2 + 50  # Y-coordinate of the horizon

    # Draw lines representing left, right, and centerline on the image
    line_image = draw_lines(
        image,
        [[
            [left_x_start, image.shape[0], left_x_horizon, horizon_y],
            [right_x_start, image.shape[0], right_x_horizon, horizon_y],
            [center_x_start, image.shape[0], center_x_end, image.shape[0] * (3 / 5)]  # Centerline
        ]],
        thickness=5,
        outer_color=[255, 0, 255]  # Color for the lines
    )
    return line_image  # Return the final image with detected lines
