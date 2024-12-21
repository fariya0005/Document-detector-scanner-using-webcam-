import cv2
import numpy as np

# Function to find the biggest contour
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimum area to consider
            peri = cv2.arcLength(contour, True)  # Perimeter
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)  # Approximate polygon
            if len(approx) == 4 and area > max_area:  # Check for quadrilateral
                biggest = approx
                max_area = area
    return biggest, max_area

# Function to reorder points of the detected contour
def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    diff = np.diff(points, axis=1)
    new_points[0] = points[np.argmin(add)]  # Top-left
    new_points[3] = points[np.argmax(add)]  # Bottom-right
    new_points[1] = points[np.argmin(diff)]  # Top-right
    new_points[2] = points[np.argmax(diff)]  # Bottom-left
    return new_points

# Function to draw a rectangle around the detected document
def drawRectangle(img, points, thickness):
    points = points.reshape(4, 2)
    for i in range(4):
        pt1 = tuple(points[i])
        pt2 = tuple(points[(i + 1) % 4])
        cv2.line(img, pt1, pt2, (0, 255, 0), thickness)
    return img

# Function to stack images for display
def stackImages(imageArray, scale, labels=[]):
    rows = len(imageArray)
    cols = len(imageArray[0])
    rowsAvailable = isinstance(imageArray[0], list)
    width = imageArray[0][0].shape[1]
    height = imageArray[0][0].shape[0]
    stackedImages = None

    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                if imageArray[x][y] is None:
                    imageArray[x][y] = np.zeros((height, width, 3), np.uint8)
                if imageArray[x][y].shape[:2] != (height, width):
                    imageArray[x][y] = cv2.resize(imageArray[x][y], (width, height))
                if len(imageArray[x][y].shape) == 2:
                    imageArray[x][y] = cv2.cvtColor(imageArray[x][y], cv2.COLOR_GRAY2BGR)
        hor = [np.hstack(imageArray[x]) for x in range(rows)]
        stackedImages = np.vstack(hor)
    else:
        for x in range(rows):
            if imageArray[x] is None:
                imageArray[x] = np.zeros((height, width, 3), np.uint8)
            if imageArray[x].shape[:2] != (height, width):
                imageArray[x] = cv2.resize(imageArray[x], (width, height))
            if len(imageArray[x].shape) == 2:
                imageArray[x] = cv2.cvtColor(imageArray[x], cv2.COLOR_GRAY2BGR)
        stackedImages = np.hstack(imageArray)

    if labels:
        each_img_width = int(stackedImages.shape[1] / cols)
        each_img_height = int(stackedImages.shape[0] / rows)
        for d in range(rows):
            for c in range(cols):
                label = labels[d][c]
                cv2.rectangle(stackedImages, (c * each_img_width, each_img_height * d),
                              (c * each_img_width + len(label) * 13 + 27, 30 + each_img_height * d),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(stackedImages, label, (each_img_width * c + 10, each_img_height * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

    return cv2.resize(stackedImages, (0, 0), None, scale, scale)
