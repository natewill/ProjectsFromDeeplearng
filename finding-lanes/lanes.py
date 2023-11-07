import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
 #########################################################




def distorts(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_intrest(image):
    height = image.shape[0]
    polygons = np.array([[(0, height-0), (524, 279), (588, 279), (1000, height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result




##########################################################



import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []
images = glob.glob('chessboard2.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

    if ret == True or ret == Falsechess:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (7,6), corners2, ret)

image = cv2.imread('test_image.jpg')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
undist = cv2.undistort(image, mtx, dist, None, mtx)



#####################################################



image_d = distorts(image)
image_d = region_of_intrest(image)


#####################################################


#(70, height-100), (70, height-50), (700, height-50), (700, height-100), (287, 180)



cap = cv2.VideoCapture('test3.mp4')
chess = 'chessboard.jpg'
frame = image_d
height = frame.shape[0]
cv2.circle(frame, (0, height), 5, (0, 0, 255), -1)
cv2.circle(frame, (1000, height), 5, (0, 0, 255), -1)
cv2.circle(frame, (588, 279), 5, (0, 0, 255), -1)
#cv2.circle(frame, (287, 180), 5, (0, 0, 255), -1)
cv2.circle(frame, (524, 279), 5, (0, 0, 255), -1)

chess_image = cv2.imread('chessboard.jpg')


pts1 = np.float32([[0, height], [524, 279], [588, 279], [1000, height]])
pts2 = np.float32([[0, 0], [800, 0], [800, 600], [0, 600]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)

result = cv2.warpPerspective(frame, matrix, (600, 600))

cv2.imshow('frame', frame)
cv2.imshow('frame2', result)

cv2.waitKey(0)



##################################################




"""
while(cap.isOpened()):
    _, frame = cap.read()
    height = frame.shape[0]
    frame = distort(frame)
    masked = region_of_intrest(frame)

    cv2.circle(frame, (70, height-50), 5, (0, 0, 255), -1)
    cv2.circle(frame, (600, height-50), 5, (0, 0, 255), -1)
    cv2.circle(frame, (350, height-200), 5, (0, 0, 255), -1)
    #cv2.circle(frame, (287, 180), 5, (0, 0, 255), -1)
    cv2.circle(frame, (300, height-200), 5, (0, 0, 255), -1)

    pts1 = np.float32([[70, height-50], [300, height-200], [350, height-200], [600, height-100]])
    pts2 = np.float32([[0, 0], [600, 0], [600, 400], [0, 400]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    result = cv2.warpPerspective(masked, matrix, (600, 600))

    cv2.imshow("Frame", rotateImage(result, 240))
    cv2.imshow("Frame2", frame)
    cv2.imshow("Frame3", masked)

    #70, height-100), (70, height-50), (700, height-50), (700, height-100), (287, 180)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
"""
