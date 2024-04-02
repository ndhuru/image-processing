import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

# Define function to create a region of interest mask on the image
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Define function to draw lines on the image
def draw_lines(img, lines, thickness=5, outer_color=[255, 0, 255]):
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )
    img = np.copy(img)
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # Skip lines with zero denominator
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) < 0.5:
                continue
            if slope <= 0:
                cv2.line(line_img, (x1, y1), (x2, y2), outer_color, thickness)
            else:
                cv2.line(line_img, (x1, y1), (x2, y2), outer_color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

# Define the main processing pipeline
def pipeline(image):
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Canny edge detection
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    # Create a region of interest
    cropped_image = region_of_interest(
        cannyed_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )
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

    # Apply Gaussian blur to grayscale image
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Calculate dynamic Canny thresholds
    median_intensity = np.median(gray_image)
    lower_threshold = int(max(0, 0.7 * median_intensity))
    upper_threshold = int(min(255, 1.3 * median_intensity))
    cannyed_image = cv2.Canny(gray_image, lower_threshold, upper_threshold)

    if lines is None:
        return image  # Return original image if no lines detected

    # Initialize lists to store points of left and right lines
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) < 0.5:
                continue
            if slope <= 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    min_y = int(image.shape[0] * (3 / 5))
    max_y = int(image.shape[0])
    if left_line_x and left_line_y:
        # Fit a polynomial to the points of the left line
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
    else:
        left_x_start = left_x_end = 0

    if right_line_x and right_line_y:
        # Fit a polynomial to the points of the right line
        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
    else:
        right_x_start = right_x_end = 0

    # Calculate centerline coordinates and angle
    if left_line_x and right_line_x:
        center_x_start = (left_x_start + right_x_start) // 2
        center_x_end = (left_x_end + right_x_end) // 2
        # Calculate angle of the centerline
        centerline_angle = np.arctan2(min_y - max_y, center_x_end - center_x_start) * 180 / np.pi
    else:
        # Default centerline to face 90 degrees if only one outer line is detected
        center_x_start = width // 2
        center_x_end = width // 2
        centerline_angle = 90

    if centerline_angle < 0:
        centerline_angle += 180  # Convert negative angles to positive range [0, 180]

    # Determine cardinal direction based on angle
    if 45 <= centerline_angle < 135:
        direction_text = "N"
    elif 135 <= centerline_angle < 225:
        direction_text = "W"
    elif 225 <= centerline_angle < 315:
        direction_text = "S"
    else:
        direction_text = "E"
    print(centerline_angle)

    # Convert image to BGR for compatibility with OpenCV
    output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Overlay text on the image
    cv2.putText(output_image, direction_text, (int(width / 2), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    horizon_y = image.shape[0] // 2 + 50
    left_x_horizon = int(poly_left(horizon_y))
    right_x_horizon = int(poly_right(horizon_y))

    # Draw lines on the image
    line_image = draw_lines(
        output_image,
        [[
            [left_x_start, max_y, left_x_horizon, horizon_y],
            [right_x_start, max_y, right_x_horizon, horizon_y],
            [center_x_start, max_y, center_x_end, min_y]  # Adding centerline
        ]],
        thickness=5,
        outer_color=[255, 0, 255]
    )
    return line_image

