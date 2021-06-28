import cv2
import math
import numpy as np

from utils import load_coefficients, save_coefficients
from tags import TAGS
from tag_detector import TagDetector


BLOCK_SIZE = 4
GRID_SHAPE = (16, 16)
GRID_SIZE = (GRID_SHAPE[0] * BLOCK_SIZE, GRID_SHAPE[1] * BLOCK_SIZE)

frame_width = 740
frame_height = 740
frame_points = np.array(
    [[0, 0], [frame_width, 0], [frame_width, frame_height], [0, frame_height]],
    dtype=np.float32,
)

margin_trbl = (20, 20, 20, 20)

gaps = (4, 4)

roiWidth = frame_width - (margin_trbl[1] + margin_trbl[3])
roiHeight = frame_height - (margin_trbl[0] + margin_trbl[2])

roiPoints = np.array(
    [
        [margin_trbl[1], margin_trbl[0]],
        [frame_width - margin_trbl[3], margin_trbl[0]],
        [frame_width - margin_trbl[3], frame_height - margin_trbl[2]],
        [margin_trbl[1], frame_height - margin_trbl[2]],
    ],
    dtype=np.float32,
)

roiToFrameH = np.matmul(
    np.array(
        [[roiWidth / frame_width, 0, 0], [0, roiHeight / frame_height, 0], [0, 0, 1]]
    ),
    cv2.findHomography(roiPoints, frame_points, cv2.LMEDS)[0],
)

gridToRoiScale = np.array(
    [[GRID_SIZE[0] / roiWidth, 0, 0], [0, GRID_SIZE[1] / roiHeight, 0], [0, 0, 1]]
)
gridMove = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])
gridToFrameH = np.matmul(gridMove, np.matmul(gridToRoiScale, roiToFrameH))

lastH = gridToFrameH

print(frame_points)
print(roiPoints)
print(lastH)


def isConvex(points):
    if len(points) < 3:
        return True
    else:  # len >= 3
        lastO = 0
        for i in range(-1, len(points) - 2):
            o = orientation(points[i], points[i + 1], points[i + 2])
            if (lastO < 0 and o > 0) or (lastO > 0 and o < 0):
                return False
            if o != 0.0:
                lastO = o
        return True


def orientation(p0, p1, p2):
    return np.cross(p1 - p0, p2 - p1)[0]


last_detected_tags = np.full(GRID_SHAPE, -1, dtype=np.int32)


def keystone(img):
    global lastH
    undistorted = cv2.undistort(img, mtx, dist, None, None)
    undistortedCopy = undistorted.copy()

    # imgray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    # imgray = cv2.fastNlMeansDenoising(imgray)

    # blockSize = round(min(imgray.shape[0],imgray.shape[1]) * 0.1)
    # blockSize = blockSize + ( 1 if blockSize % 2 == 0 else 0 )

    # thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #            cv2.THRESH_BINARY_INV,blockSize,5)

    # filters image bilaterally and displays it
    bilatImg = cv2.bilateralFilter(undistorted, 5, 175, 175)

    # finds edges of bilaterally filtered image and displays it
    edgeImg = cv2.Canny(bilatImg, 75, 200)

    kernel = np.ones((3, 3), "uint8")
    edgeImg = cv2.dilate(edgeImg, kernel)
    edgeImg = cv2.erode(edgeImg, kernel)

    # cv2.imshow('edges', edgeImg)

    contours, hierarchy = cv2.findContours(
        edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # cv2.drawContours(undistorted, contours, -1, (255,0,0), 3)

    imgArea = edgeImg.shape[0] * edgeImg.shape[1]

    trapezoidContours = []
    trapezoidCorners = []

    for i, contour in enumerate(contours):
        contourArea = cv2.contourArea(contour)
        if contourArea > 0.0 * imgArea and contourArea < 1.0 * imgArea:
            # find the perimeter of the first closed contour
            perim = cv2.arcLength(contour, True)
            # setting the precision
            epsilon = 0.1 * perim
            # approximating the contour with a polygon
            approxCorners = cv2.approxPolyDP(contour, epsilon, True)
            # check how many vertices has the approximate polygon
            approxCornersNumber = len(approxCorners)

            # if approxCornersNumber > 0:
            # cv2.drawContours(undistorted, contours, i, (255,255,0), 3)

            if approxCornersNumber == 4:
                # cv2.drawContours(undistorted, contours, i, (0,255,0), 3)
                if isConvex(approxCorners):
                    if (
                        orientation(
                            approxCorners[0], approxCorners[1], approxCorners[2]
                        )
                        < 0
                    ):
                        # points are CCW
                        approxCorners = np.flip(approxCorners, axis=0)

                    topLeft = min(approxCorners, key=np.linalg.norm)
                    while not np.array_equal(approxCorners[0], topLeft):
                        approxCorners = np.roll(approxCorners, 1)
                    trapezoidContours.append(contour)
                    trapezoidCorners.append(approxCorners)

    w = GRID_SIZE[0]
    h = GRID_SIZE[1]
    if len(trapezoidContours) >= 1:
        sourcePoints = trapezoidCorners[0]

        #        print(sourcePoints, targetPoints)
        cv2.drawContours(undistorted, trapezoidContours, 0, (0, 0, 255), 3)

        H = cv2.findHomography(sourcePoints, frame_points, cv2.LMEDS)[0]
        lastH = H

        scale_factor = max(undistorted.shape[0], undistorted.shape[1]) / min(
            frame_width, frame_height
        )
        scale = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
        keytoned1 = cv2.warpPerspective(
            undistortedCopy,
            np.matmul(scale, H),
            (
                math.ceil(frame_width * scale_factor),
                math.ceil(frame_height * scale_factor),
            ),
            flags=cv2.INTER_AREA,
        )

        for i in range(len(roiPoints)):
            cv2.circle(
                keytoned1, tuple(roiPoints[i] * scale_factor), 2, (0, 0, 255), -1
            )
        cv2.imshow("frame", keytoned1)

        scale_factor = max(undistorted.shape[0], undistorted.shape[1]) / min(
            frame_width, frame_height
        )
        scale = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
        keystoned2 = cv2.warpPerspective(
            undistortedCopy,
            np.matmul(np.matmul(scale, roiToFrameH), H),
            (math.ceil(roiWidth * scale_factor), math.ceil(roiHeight * scale_factor)),
            flags=cv2.INTER_AREA,
        )

        cellSize = keystoned2.shape[0] / GRID_SIZE[0]
        for i in range(GRID_SIZE[0]):
            cv2.line(
                keystoned2,
                (0, math.ceil((i + 0.5) * cellSize)),
                (keystoned2.shape[0], math.ceil((i + 0.5) * cellSize)),
                (0, 255, 0),
                1,
            )
        for i in range(GRID_SIZE[1]):
            cv2.line(
                keystoned2,
                (math.ceil((i + 0.5) * cellSize), 0),
                (math.ceil((i + 0.5) * cellSize), keystoned2.shape[1]),
                (0, 255, 0),
                1,
            )

        cv2.imshow("region of interest", keystoned2)

        keystoned2_gray = cv2.cvtColor(keystoned2, cv2.COLOR_BGR2GRAY)

        global last_detected_tags
        tiles = tag_detector.extract_tiles(keystoned2_gray)
        detected_tags = tag_detector.detect_tags(tiles)
        if not np.array_equal(last_detected_tags, detected_tags):
            print("new tags:\n", detected_tags)
            last_detected_tags = detected_tags

        grid_img = tiles.reshape(GRID_SIZE)
        grid_img_big = cv2.resize(
            grid_img * 255,
            (7 * grid_img.shape[0], 7 * grid_img.shape[1]),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("grid_img", grid_img_big)

        scale_factor = 7
        scale = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
        keystoned3 = cv2.warpPerspective(
            undistortedCopy,
            np.matmul(scale, np.matmul(gridToFrameH, H)),
            (
                math.ceil(GRID_SIZE[0] * scale_factor),
                math.ceil(GRID_SIZE[1] * scale_factor),
            ),
            flags=cv2.INTER_AREA,
        )

        cellSize = keystoned3.shape[0] / GRID_SIZE[0]
        for i in range(GRID_SIZE[0]):
            cv2.line(
                keystoned3,
                (0, math.ceil((i + 0.5) * cellSize)),
                (keystoned3.shape[0], math.ceil((i + 0.5) * cellSize)),
                (0, 255, 0),
                1,
            )
        for i in range(GRID_SIZE[1]):
            cv2.line(
                keystoned3,
                (math.ceil((i + 0.5) * cellSize), 0),
                (math.ceil((i + 0.5) * cellSize), keystoned3.shape[1]),
                (0, 255, 0),
                1,
            )

        cv2.imshow("region of interest with grid", keystoned3)

        keystoned = cv2.warpPerspective(
            undistortedCopy, np.matmul(gridToFrameH, H), (w, h), flags=cv2.INTER_AREA
        )
        keystoned = cv2.cvtColor(keystoned, cv2.COLOR_BGR2GRAY)
        ret, keystoned = cv2.threshold(
            keystoned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        keystonedBig = cv2.resize(
            keystoned,
            (7 * keystoned.shape[0], 7 * keystoned.shape[1]),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("detected pattern", keystonedBig)

    #    cv2.resizeWindow('keystoned', dstD.shape[0]*10, dstD.shape[1]*10)

    for i in range(len(roiPoints)):
        p = np.append(roiPoints[i], 1)
        pHInv = np.matmul(np.linalg.inv(lastH), p)
        if pHInv[2] != 0.0:
            pHInvTuple = (int(pHInv[0] / pHInv[2]), int(pHInv[1] / pHInv[2]))
            cv2.circle(undistorted, pHInvTuple, 2, (0, 0, 255), -1)

    cv2.imshow("keystone", undistorted)


mtx, dist = load_coefficients("camera-profile.yml")


def from_camera():
    capture = cv2.VideoCapture(0)

    while True:
        ret, src = capture.read()
        keystone(src)
        key = cv2.waitKey(1)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


def from_file():
    src = cv2.imread("snapshot.jpg")
    keystone(src)
    cv2.waitKey()


tag_detector = TagDetector(
    GRID_SHAPE,
    (BLOCK_SIZE, BLOCK_SIZE),
    (gaps[0] / roiHeight, gaps[1] / roiWidth),
    TAGS,
)

from_file()
