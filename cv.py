import cv2
import numpy as np

def preprocessing(img):
    # 리사이징 (필요한 경우)
    img = cv2.resize(img, (500, 500))
    
    # 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 이진화 적용
    _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 모폴로지 연산 (침식 후 팽창 - 열림 연산)
    kernel = np.ones((5,5), np.uint8)
    morph = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    return morph

# 객체 이미지 불러오기
obj = cv2.imread('1.jpg', cv2.IMREAD_COLOR)
obj_processed = preprocessing(obj)
obj_contours, _ = cv2.findContours(obj_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
obj_pts = obj_contours[0]

# 카메라 연결
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("camera open failed")
    exit()

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (500, 500))
    if not ret:
        print("Can't read camera")
        break

    cv2.imshow('PC_camera', img)

    key = cv2.waitKey(1)
    if key == ord('a'):
        break
    elif key == ord('q'):  # 'q' 키를 누르면 종료
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
img_processed = preprocessing(img)
contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 결과 영상
dst = img.copy()

# 입력 영상의 모든 객체 영역에 대해서
for pts in contours:
    if cv2.contourArea(pts) < 1000:
        continue

    rc = cv2.boundingRect(pts)
    cv2.rectangle(dst, rc, (255, 0, 0), 1)

    # 모양 비교
    dist = cv2.matchShapes(obj_pts, pts, cv2.CONTOURS_MATCH_I3, 0)  # dist 값으로 유사도 측정

    cv2.putText(dst, str(round(dist, 4)), (rc[0], rc[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

    if dist < 0.1:  # dist 값이 작을수록 obj와 유사
        cv2.rectangle(dst, rc, (0, 0, 255), 2)

obj = cv2.resize(obj, (500, 500))
cv2.imshow('obj', obj)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
