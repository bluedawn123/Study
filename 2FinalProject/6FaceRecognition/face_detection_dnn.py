import cv2
import numpy as np

model_name = './2FinalProject/data/res10_300x300_ssd_iter_140000.caffemodel' #resnet사용. 300x300크기. 메타정보 학습값 = caffemodel.
prototxt_name = './2FinalProject/data/deploy.prototxt.txt'  #모델의 설계도. 메타정보(shape, layer의 구성저장)
min_confidence = 0.3
file_name = './2FinalProject/data/beckhamFamily.jpg'


#detection 이미지 보여주기
def detectAndDisplay(frame):
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name) #readnet함수를 이용해 model생성
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)) #4가지의 파라미터. #dnn모델 생성

        #1. 이미지 크기 300,300사용하므로, 300,300으로. 2. 이미지 크기 비율 = 1.0(변형 없음)
        #3. cnn에서 사용할 이미지 -> 300, 300          4번의 파라미터는 경험치.. 가장 많이 쓰는 경우
    model.setInput(blob)
    detections = model.forward()  #detection 결과값을 4차원 배열로 저장된다. 
   
    for i in range(0, detections.shape[2]):  #2차원엔 0,0만 사용할것. 최대의 박스크기 의미(200)

            #얼굴일 확률 보여주기
            confidence = detections[0, 0, i, 2]  #detection은 4차원인데, i가 얼굴일 확률.
            if confidence > min_confidence:
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    print(i, confidence,detections[0, 0, i, 3], startX, startY, endX, endY)
     
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("DNN을 사용한 얼굴 인식", frame)
    
img = cv2.imread(file_name)
(height, width) = img.shape[:2]
#원본 보여주기
cv2.imshow("Original Image", img)  
detectAndDisplay(img)
cv2.waitKey(0)  #아무키 입력시 종료
cv2.destroyAllWindows()
