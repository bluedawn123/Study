import numpy as np
import dlib
import cv2

#68개의 점 배정
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 68))
EYES = list(range(36, 48))

predictor_file = './2FinalProject/data/shape_predictor_68_face_landmarks.dat' 
image_file = './2FinalProject/data/marathon_04.jpg'

detector = dlib.get_frontal_face_detector()    
predictor = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #노이즈를 줄이기 위해(인식률증가) 회색

rects = detector(gray, 1) #detection 전 이미지 레이어를 1번 적용

#print("Number of faces detected: {}".format(len(rects)))


for (i, rect) in enumerate(rects):
    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])  #predicor ->파트 분류 후 p(점)을 x, y좌표로 배열생성
    show_parts = points[ALL]  #모든 점 갖고오기. 2차원 배열로 저장
    
    #print(show_parts) #점의 좌표 출력

    for (i, point) in enumerate(show_parts):  #i ->점의 순서, point ->좌표값
        x = point[0,0] #x좌표  [[  , ]]
        y = point[0,1] #y좌표
      
      
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)   #원 그리기(이미지, (좌표), 1, (색), 다 채우기
        cv2.putText(image, "{}".format(i + 1), (x, y - 2),#점의 번호를 text순서화
		cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)    #폰트

cv2.imshow("Face Landmark", image)
cv2.waitKey(0)   
