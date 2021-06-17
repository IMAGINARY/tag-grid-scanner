import cv2
import math
import numpy as np

from utils import load_coefficients, save_coefficients

BLOCK_SIZE = 4
GRID_SIZE = (16*BLOCK_SIZE, 16*BLOCK_SIZE)

frameWidth = 15.45
frameHeight = 15.45
framePoints = np.array([[0, 0],[frameWidth, 0],[frameWidth, frameHeight],[0, frameHeight]],dtype=np.float32)

marginTRBL = (0.84, 0.84, 0.84, 0.84)
roiWidth = frameWidth-(marginTRBL[1]+marginTRBL[3])
roiHeight = frameHeight-(marginTRBL[0]+marginTRBL[2])

roiPoints = np.array([
    [marginTRBL[1], marginTRBL[0]],
    [frameWidth-marginTRBL[3], marginTRBL[0]],
    [frameWidth-marginTRBL[3], frameHeight-marginTRBL[2]],
    [marginTRBL[1], frameHeight-marginTRBL[2]],
],dtype=np.float32)

roiToFrameH = np.matmul(
    np.array([[roiWidth/frameWidth,0,0],[0,roiHeight/frameHeight,0],[0,0,1]]),
    cv2.findHomography(roiPoints, framePoints,cv2.LMEDS)[0]
    )

gridToRoiScale = np.array([[GRID_SIZE[0]/roiWidth,0,0],[0,GRID_SIZE[1]/roiHeight,0],[0,0,1]])
gridMove = np.array([[1,0,-0.5],[0,1,-0.5],[0,0,1]])
gridToFrameH = np.matmul(gridMove,np.matmul(gridToRoiScale,roiToFrameH))

lastH = gridToFrameH

print(framePoints)
print(roiPoints)
print(lastH)

def isConvex(points):
    if len(points) < 3:
        return True
    else: # len >= 3
        lastO = 0
        for i in range(-1, len(points)-2):
            o = orientation(points[i],points[i+1],points[i+2])
            if (lastO < 0 and o > 0) or (lastO > 0 and o < 0):
                return False
            if o != 0.0:
                lastO = o
        return True

def orientation(p0, p1, p2):
    return np.cross(p1 - p0, p2 - p1)[0]

def keystone(img):
    global lastH
    undistorted = cv2.undistort(img, mtx, dist, None, None)
    undistortedCopy = undistorted.copy()

    #imgray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    #imgray = cv2.fastNlMeansDenoising(imgray)

    #blockSize = round(min(imgray.shape[0],imgray.shape[1]) * 0.1)
    #blockSize = blockSize + ( 1 if blockSize % 2 == 0 else 0 )

    #thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #            cv2.THRESH_BINARY_INV,blockSize,5)


    #filters image bilaterally and displays it
    bilatImg = cv2.bilateralFilter(undistorted, 5, 175, 175)

    #finds edges of bilaterally filtered image and displays it
    edgeImg = cv2.Canny(bilatImg, 75, 200)
    
    kernel = np.ones((3, 3), 'uint8')
    edgeImg = cv2.dilate(edgeImg, kernel)
    edgeImg = cv2.erode(edgeImg, kernel)

    #cv2.imshow('edges', edgeImg)

    contours, hierarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    #cv2.drawContours(undistorted, contours, -1, (255,0,0), 3)

    imgArea = edgeImg.shape[0] * edgeImg.shape[1]

    trapezoidContours = []
    trapezoidCorners = []

    for i, contour in enumerate(contours):
        contourArea = cv2.contourArea(contour)
        if contourArea > 0.0 * imgArea and contourArea < 1.0 * imgArea:
            # find the perimeter of the first closed contour
            perim = cv2.arcLength(contour, True)
            # setting the precision
            epsilon = 0.1*perim
            # approximating the contour with a polygon
            approxCorners = cv2.approxPolyDP(contour, epsilon, True)
            # check how many vertices has the approximate polygon
            approxCornersNumber = len(approxCorners)
            
            #if approxCornersNumber > 0:
                #cv2.drawContours(undistorted, contours, i, (255,255,0), 3)
            
            if approxCornersNumber == 4:
                #cv2.drawContours(undistorted, contours, i, (0,255,0), 3)
                if isConvex(approxCorners):
                    if orientation(approxCorners[0],approxCorners[1],approxCorners[2]) < 0:
                        # points are CCW
                        approxCorners = np.flip(approxCorners, axis=0)
                    
                    topLeft = min(approxCorners, key = np.linalg.norm)
                    while(not np.array_equal(approxCorners[0], topLeft)):
                        approxCorners = np.roll(approxCorners, 1)
                    trapezoidContours.append(contour)
                    trapezoidCorners.append(approxCorners)

    w = GRID_SIZE[0]
    h = GRID_SIZE[1]
    if len(trapezoidContours) >= 1:
        sourcePoints = trapezoidCorners[0]

#        print(sourcePoints, targetPoints)
        cv2.drawContours(undistorted, trapezoidContours, 0, (0,0,255), 3)

        H = cv2.findHomography(sourcePoints,framePoints,cv2.LMEDS)[0]
        lastH = H

        scaleFactor = 30;
        scale = np.array([[scaleFactor,0,0],[0,scaleFactor,0],[0,0,1]])
        keytoned1 = cv2.warpPerspective(undistortedCopy,np.matmul(scale,H),(math.ceil(frameWidth*scaleFactor),math.ceil(frameHeight*scaleFactor)),flags=cv2.INTER_AREA)
        
        for i in range(len(roiPoints)):
            cv2.circle(keytoned1, tuple(roiPoints[i] * scaleFactor), 2, (0,0,255), -1)
        cv2.imshow('frame', keytoned1 )

        scaleFactor = 30;
        scale = np.array([[scaleFactor,0,0],[0,scaleFactor,0],[0,0,1]])
        keystoned2 = cv2.warpPerspective(undistortedCopy,np.matmul(np.matmul(scale,roiToFrameH),H),(math.ceil(roiWidth*scaleFactor),math.ceil(roiHeight*scaleFactor)),flags=cv2.INTER_AREA)
        
        cellSize = keystoned2.shape[0] / GRID_SIZE[0]
        for i in range(GRID_SIZE[0]):
            cv2.line(keystoned2,(0,math.ceil((i+0.5) * cellSize)),(keystoned2.shape[0],math.ceil((i+0.5) * cellSize)),(0,255,0),1)
        for i in range(GRID_SIZE[1]):
            cv2.line(keystoned2,(math.ceil((i+0.5) * cellSize),0),(math.ceil((i+0.5) * cellSize),keystoned2.shape[1]),(0,255,0),1)
        
        cv2.imshow('region of interest', keystoned2)


        scaleFactor = 7;
        scale = np.array([[scaleFactor,0,0],[0,scaleFactor,0],[0,0,1]])
        keystoned3 = cv2.warpPerspective(undistortedCopy,np.matmul(scale,np.matmul(gridToFrameH,H)),(math.ceil(GRID_SIZE[0]*scaleFactor),math.ceil(GRID_SIZE[1]*scaleFactor)),flags=cv2.INTER_AREA)
        
        cellSize = keystoned3.shape[0] / GRID_SIZE[0]
        for i in range(GRID_SIZE[0]):
            cv2.line(keystoned3,(0,math.ceil((i+0.5) * cellSize)),(keystoned3.shape[0],math.ceil((i+0.5) * cellSize)),(0,255,0),1)
        for i in range(GRID_SIZE[1]):
            cv2.line(keystoned3,(math.ceil((i+0.5) * cellSize),0),(math.ceil((i+0.5) * cellSize),keystoned3.shape[1]),(0,255,0),1)
        
        cv2.imshow('region of interest with grid', keystoned3)

        keystoned = cv2.warpPerspective(undistortedCopy,np.matmul(gridToFrameH,H),(w, h),flags=cv2.INTER_AREA)    
        keystoned = cv2.cvtColor(keystoned, cv2.COLOR_BGR2GRAY)
        ret, keystoned = cv2.threshold(keystoned,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        keystonedBig = cv2.resize( keystoned,(7 * keystoned.shape[0],7 * keystoned.shape[1]), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('detected pattern', keystonedBig)

#    cv2.resizeWindow('keystoned', dstD.shape[0]*10, dstD.shape[1]*10)

    
    for i in range(len(roiPoints)):
        p = np.append(roiPoints[i],1)
        pHInv = np.matmul(np.linalg.inv(lastH),p)
        if pHInv[2] != 0.0:
            pHInvTuple = (int(pHInv[0]/pHInv[2]),int(pHInv[1]/pHInv[2]))
            cv2.circle(undistorted, pHInvTuple, 2, (0,0,255), -1)
        
    cv2.imshow('keystone', undistorted)

mtx, dist = load_coefficients('camera-profile.yml')

capture = cv2.VideoCapture(0)
#src = cv2.imread('snapshot.jpg')

while(True):
    ret, src = capture.read()
    keystone(src)
    key = cv2.waitKey(1)
    if key == 27:
        break;

capture.release()
cv2.destroyAllWindows()
