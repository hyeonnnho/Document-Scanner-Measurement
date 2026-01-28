import cv2
import numpy as np

###################################
widthImg = 640
heightImg = 480
###################################

cap = cv2.VideoCapture(0)
cap.set(3, widthImg)
cap.set(4, heightImg)
cap.set(10, 150)

# 트랙바 설정
def empty(a): pass
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 100, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 200, 255, empty)
cv2.createTrackbar("Area", "Parameters", 5000, 30000, empty)

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    minArea = cv2.getTrackbarPos("Area", "Parameters")
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    
    # 깔끔하게 가장자리 10px씩 잘라내기
    imgCropped = imgOutput[10:imgOutput.shape[0]-10, 10:imgOutput.shape[1]-10]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))
    return imgCropped

# ★ [추가된 함수] cm 단위 그리드 그리기
def drawGrid(img):
    # A4 너비: 210mm, 높이: 297mm
    # 현재 이미지 너비(widthImg)가 210mm에 해당함.
    # 10mm(1cm)가 몇 픽셀인지 계산
    pixel_per_cm = widthImg / 21.0 
    
    # 가로선 그리기 (1cm 간격)
    for i in range(1, 30): # 높이가 약 29.7cm이므로
        y = int(i * pixel_per_cm)
        cv2.line(img, (0, y), (widthImg, y), (255, 0, 255), 1) # 보라색 선
        
    # 세로선 그리기 (1cm 간격)
    for i in range(1, 21): # 너비가 21cm이므로
        x = int(i * pixel_per_cm)
        cv2.line(img, (x, 0), (x, heightImg), (255, 0, 255), 1)

    return img

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    t1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    t2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgBlur, t1, t2)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    
    biggest = getContours(imgThres)
    
    if biggest.size != 0:
        cv2.drawContours(imgContour, [biggest], -1, (0, 255, 0), 20)
        imgWarped = getWarp(img, biggest)
        
        # ★ 변환된 이미지 위에 1cm 격자 그리기
        imgGrid = drawGrid(imgWarped.copy())
        
        cv2.imshow("Smart Scanner (cm Grid)", imgGrid)
    else:
        cv2.imshow("Smart Scanner (cm Grid)", img)

    cv2.imshow("Original", imgContour)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()