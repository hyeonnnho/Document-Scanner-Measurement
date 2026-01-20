import cv2
import numpy as np

# 1. 설정값
widthImg = 640
heightImg = 480
cap = cv2.VideoCapture(0)
cap.set(3, widthImg)
cap.set(4, heightImg)
cap.set(10, 150)

# --- [핵심 함수: 가장 큰 사각형 찾기] ---
def getContours(img):
    # 가장 큰 사각형의 꼭짓점(4개)을 저장할 변수
    biggest = np.array([])
    maxArea = 0
    
    # 외곽선(Contours) 찾기
    # RETR_EXTERNAL: 바깥쪽 외곽선만 찾음 (내부 구멍은 무시)
    # CHAIN_APPROX_NONE: 모든 점의 좌표를 저장
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt) # 면적 계산
        
        if area > 5000: # 너무 작은 노이즈(면적 5000 미만)는 무시
            # 둘레(Perimeter) 길이 계산
            peri = cv2.arcLength(cnt, True)
            
            # ★ 다각형 근사 (Douglas-Peucker 알고리즘)
            # 0.02 * peri: 오차 허용 범위(epsilon). 둘레의 2% 오차범위 내에서 직선으로 근사
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # 꼭짓점이 4개이고(사각형), 현재까지 찾은 것 중 가장 넓다면?
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    
    # 찾은 가장 큰 사각형의 꼭짓점 좌표 반환, 없으면 빈 배열 반환
    return biggest

while True:
    success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg)) # 크기 고정
    imgContour = img.copy() # 결과를 그릴 복사본 이미지
    
    # [전처리]
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    
    # [가장 큰 사각형 찾기]
    biggest = getContours(imgCanny)
    
    # [시각화] 찾은 사각형이 있다면 초록색 선으로 그리기
    if biggest.size != 0:
        # drawContours: 찾은 좌표(biggest)를 이미지 위에 그림
        cv2.drawContours(imgContour, [biggest], -1, (0, 255, 0), 20)
    
    # 결과 출력
    cv2.imshow("Result", imgContour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()