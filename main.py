import cv2
import numpy as np


def add_text(img):
    _, width, _ = img.shape
    cv2.putText(img, "Original", (int(0.5), 22), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
    cv2.putText(img, "Gray", (int(width*.25 + 0.5), 22), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
    cv2.putText(img, "Bitwise", (int(width*.5 + 0.5), 22), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
    cv2.putText(img, "Contours", (int(width*.75 + 0.5), 22), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
    return img


def find_pupil(img):
    # Binarize and invert image since white object (white pupil) is of interest
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bit = cv2.bitwise_not(img_gray)
    # Setting adequate threshold values is essential for a clean detection
    _, img_binary = cv2.threshold(img_bit, 210, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # TODO: Use moments of HU to filter for main circle
    pass

    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 2)

    img_comparison = np.hstack((img,
                                cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR),
                                cv2.cvtColor(img_bit, cv2.COLOR_GRAY2BGR),
                                img_contours))
    add_text(img_comparison)
    cv2.imshow("images", img_comparison)
    cv2.waitKey(0)

    # TODO: Get center of contour
    position = [0, 0]
    return position


if __name__ == "__main__":
    img_file = "samples/sample1.png"
    img = cv2.imread(img_file)
    find_pupil(img)
