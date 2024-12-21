import cv2
import numpy as np
MIN_CONTOUR_AREA = 1000
ASPECT_RATIO_RANGE = (0.5, 2.0)
BLUR_KERNEL_SIZE = (5, 5)
EDGE_KERNEL_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCKSIZE = 11
ADAPTIVE_THRESH_C = 2
EDGE_THRESH_LOW = 50
EDGE_THRESH_HIGH = 150

def setup_camera():
    """
    Set up the camera and return the video capture object.
    """
    for i in range(5):  # Check from camera index 0 to 4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera at index {i} is accessible.")
            return cap
        else:
            print(f"Camera at index {i} is not accessible.")
    print("No camera found! Exiting...")
    exit(0)

def detect_document(img):
    """
    Detect the largest quadrilateral contour resembling a document.
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.GaussianBlur(imgGray, BLUR_KERNEL_SIZE, 0)
    imgThresh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, ADAPTIVE_THRESH_BLOCKSIZE, ADAPTIVE_THRESH_C)
    imgEdges = cv2.Canny(imgThresh, EDGE_THRESH_LOW, EDGE_THRESH_HIGH)
    kernel = np.ones(EDGE_KERNEL_SIZE, np.uint8)
    imgEdges = cv2.dilate(imgEdges, kernel, iterations=2)
    imgEdges = cv2.erode(imgEdges, kernel, iterations=1)
    contours, _ = cv2.findContours(imgEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_CONTOUR_AREA:  
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:  
                rect = cv2.boundingRect(approx)
                aspect_ratio = float(rect[2]) / rect[3]  
                if ASPECT_RATIO_RANGE[0] < aspect_ratio < ASPECT_RATIO_RANGE[1]: 
                    best_contour = approx
                    max_area = area

    return best_contour, imgThresh, imgEdges

def warp_document(img, contour):
    """
    Warp the document area to produce a top-down view.
    """
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] 
    rect[2] = pts[np.argmax(s)]  
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  
    rect[3] = pts[np.argmax(diff)]  
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))
    return warp

def crop_document(image_path):
    """
    Allow the user to manually crop the document using a GUI.
    """
    img = cv2.imread(image_path)
    r = cv2.selectROI("Crop Document", img)
    if r != (0, 0, 0, 0):
        cropped_img = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        cropped_path = image_path.replace(".jpg", "_cropped.jpg")
        cv2.imwrite(cropped_path, cropped_img)
        print(f"Cropped document saved as {cropped_path}")
    else:
        print("No cropping performed.")
    cv2.destroyAllWindows()

def main():
    cap = setup_camera()
    cap.set(10, 160)  
    widthImg, heightImg = 480, 640
    print("Press SPACE to save the document, or Q to quit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture an image. Exiting...")
            break

        img = cv2.flip(img, 1)  
        img = cv2.resize(img, (widthImg, heightImg))
        best_contour, imgThresh, imgEdges = detect_document(img)

        if best_contour is not None:
            cv2.drawContours(img, [best_contour], -1, (0, 255, 0), 3)
            cv2.putText(img, "Document Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "No Document Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Webcam Feed", img)
        cv2.imshow("Thresholded Image", imgThresh)
        cv2.imshow("Edge Detection", imgEdges)

        key = cv2.waitKey(1)
        if key == ord('q'):  
            print("Exiting...")
            break
        elif key == 32 and best_contour is not None:  
            warp = warp_document(img, best_contour)
            save_path = "document.jpg"
            cv2.imwrite(save_path, warp)
            print(f"Document saved as {save_path}")
            crop_document(save_path)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 

