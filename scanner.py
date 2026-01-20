import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(3, 480)
cap.set(10, 150)

print("Start Scanning... 'q'를 누르면 종료됩니다.")

while True:
    success, img = cap.read()

    if not success:
        print("카메라를 찾을 수 없습니다.")
        break

    # 흑백 변환: 3차원 배열(RGB)을 2차원(밝기값)으로 차원 축소
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 가우시안 블러: 노이즈를 제거하여 엉뚱한 곳을 에지로 인식하는 것을 방지
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1) # (5,5) -> 커널(필터)의 크기
    # 캐니 에지 검출: 픽셀값의 변화량이 큰 부분을 찾음
    imgCanny = cv2.Canny(imgBlur, 200, 200) # 200: 임계값1, 200: 임계값2 -> 조절 시 민감도 변함

    cv2.imshow("Original", img)
    cv2.imshow("Canny Edge", imgCanny)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

