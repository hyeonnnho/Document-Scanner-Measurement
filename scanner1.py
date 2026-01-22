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

# 트랙바 설정 (이전 설정값 기억해서 수정하셔도 됩니다)
def empty(a): pass
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 100, 255, empty) # 추천: 80~120
cv2.createTrackbar("Threshold2", "Parameters", 200, 255, empty) # 추천: 200~255
cv2.createTrackbar("Area", "Parameters", 5000, 30000, empty)

# --- [함수 1: 윤곽선 찾기] ---
def getContours(img):
    biggest = np.array([])
    maxArea = 0
    minArea = cv2.getTrackbarPos("Area", "Parameters")
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
                
    # 결과: (4, 1, 2) 형태의 배열 [[x,y], [x,y], [x,y], [x,y]]
    return biggest

# --- [함수 2: 점 재정렬 (Linear Algebra)] ---
def reorder(myPoints):
    # myPoints의 형태를 (4, 1, 2) -> (4, 2)로 단순화
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    
    # 1. 덧셈 (Sum) -> TL, BR 찾기
    add = myPoints.sum(1) # axis=1 (x+y)
    myPointsNew[0] = myPoints[np.argmin(add)] # 합이 최소 -> Top-Left
    myPointsNew[3] = myPoints[np.argmax(add)] # 합이 최대 -> Bottom-Right
    
    # 2. 뺄셈 (Diff) -> TR, BL 찾기
    diff = np.diff(myPoints, axis=1) # (y-x) or (x-y)
    myPointsNew[1] = myPoints[np.argmin(diff)] # 차이가 최소(음수최대) -> Top-Right
    myPointsNew[2] = myPoints[np.argmax(diff)] # 차이가 최대 -> Bottom-Left
    
    return myPointsNew

# --- [함수 3: 투영 변환 (Perspective Transform)] ---
def getWarp(img, biggest):
    # 1. 점 정렬
    biggest = reorder(biggest)
    
    # 2. 변환할 4개의 점 (Source)
    pts1 = np.float32(biggest)
    
    # 3. 변환될 위치의 4개의 점 (Destination) -> 꽉 찬 직사각형
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    
    # 4. 변환 행렬(Matrix) 계산 (3x3 행렬)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    # 5. 이미지 변환 적용
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    
    # 자른 이미지에서 가장자리의 노이즈를 살짝 잘라냄 (Crop) - 5% 정도
    imgCropped = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))
    
    return imgCropped

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()
    
    # 전처리
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    t1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    t2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgBlur, t1, t2)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    
    # 사각형 찾기
    biggest = getContours(imgThres)
    
    if biggest.size != 0:
        # 찾았으면 윤곽선 그리고
        cv2.drawContours(imgContour, [biggest], -1, (0, 255, 0), 20)
        
        # ★ 투영 변환 수행!
        imgWarped = getWarp(img, biggest)
        
        # 결과창에 '변환된 문서' 보여주기
        cv2.imshow("ImageWarped", imgWarped)
    else:
        # 못 찾았으면 그냥 원본 보여주기 (에러 방지)
        cv2.imshow("ImageWarped", img)

    cv2.imshow("Workflow", imgContour) # 인식 과정
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()