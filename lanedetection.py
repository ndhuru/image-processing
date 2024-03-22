import cv2
import numpy as np
import math

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

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

def pipeline(image):
    height, width, _ = image.shape
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(
        cannyed_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )
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
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
        left_x_horizon = int(poly_left(max_y))  # Move this line inside the if block
    else:
        poly_left = None  # Define poly_left as None when no left lines are detected
        left_x_start = left_x_end = 0
        left_x_horizon = 0  # Handle the case where poly_left is not defined

    if right_line_x and right_line_y:
        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
        right_x_horizon = int(poly_right(max_y))  # Move this line inside the if block
    else:
        poly_right = None  # Define poly_right as None when no right lines are detected
        right_x_start = right_x_end = 0
        right_x_horizon = 0  # Handle the case where poly_right is not defined

    center_x_start = (left_x_start + right_x_start) // 2
    center_x_end = (left_x_end + right_x_end) // 2

    horizon_y = image.shape[0] // 2 + 50

    line_image = draw_lines(
        image,
        [[
            [left_x_start, max_y, left_x_horizon, horizon_y],
            [right_x_start, max_y, right_x_horizon, horizon_y],
            [center_x_start, max_y, center_x_end, min_y]  # Adding centerline
        ]],
        thickness=5,
        outer_color=[255, 0, 255]
    )
    return line_image
