import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np


def order_points(pts):
    """Function for ordering coordinates

    Args:
        pts (list): a list of four points specifying the (x, y) coordinates
        of each point of the rectangle.

    Returns:
        numpy.ndarray: numpy array of ordered coordinates
    """
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    """
    Applies a four-point perspective transform to the given image.

    This function transforms the input image to obtain a top-down,
    "bird's eye view" perspective of the region defined by the
    four input points.

    Args:
        image (numpy.ndarray): The input image to be transformed.
        pts (numpy.ndarray): A 4x2 array of points representing the
                             coordinates of the region to be transformed.
                             The points should be in the following order:
                             top-left, top-right, bottom-right, bottom-left.

    Returns:
        numpy.ndarray: The warped image resulting from the perspective transform.
    """
    # obtain a consistent order of the points and unpack them
    # individually

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    return edged


def detect_contours(edged_image):
    cnts = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    return screenCnt


image = cv2.imread("test_image.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()

image = imutils.resize(image, height=500)
edged = detect_edges(image)
contours = detect_contours(edged)
image_contours = cv2.polylines(
    image.copy(), [contours], isClosed=True, color=(0, 255, 0), thickness=2
)

plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.title("Edged Image"), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB))
plt.title("Image with Contours"), plt.xticks([]), plt.yticks([])
plt.show()
