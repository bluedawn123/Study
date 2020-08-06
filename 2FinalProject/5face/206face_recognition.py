import cv2
import face_recognition
import pickle
import time

image_file = "./2FinalProject/data/soccer_03.jpg"
encoding_file = 'encodings.pickle'
unknown_name = 'Unknown'
# Either cnn  or hog. The CNN method is more accurate but slower. HOG is faster but less accurate.
model_method = 'cnn'

def detectAndDisplay(image):
    start_time = time.time()  #cnn에서는 시간이 중요하므로 시간측정.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,
        model=model_method)
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    #인코딩 된 얼굴들이 맞냐, 아니냐
    for encoding in encodings:  #인코딩한 얼굴들을 for문으로 돌리게 된다.

        matches = face_recognition.compare_faces(data["encodings"], encoding) #encodings값과 encoding을 비교한다. 
        name = unknown_name #안맞는 경우도 생성

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b] #매치된 경우, 그것들을 배열에 넣음.
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]                 #어떤이름으로 매치가 되었는지
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)          #가장 많이 나온 것으로 선정하기 위해서.
        
        names.append(name) #맞는것이 없으면, unknown으로 된다.
        
    #박스 세부화
    for ((top, right, bottom, left), name) in zip(boxes, names):  #박스와, 이름 출력
        y = top - 15 if top - 15 > 15 else top + 15               #top - 15 : 얼굴만 나오게 하려고. 
       
        color = (0, 255, 0)  #B, G, R 즉, 초록색
        line = 2             #라인 크기
      
        if(name == unknown_name): #만약, 이름이 unknown이라면, 빨간색과 라인크기 1로 출력
            color = (0, 0, 255)
            line = 1
            name = ''
            
        cv2.rectangle(image, (left, top), (right, bottom), color, line)  #사각형
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,    #텍스트설정-> 폰트, 이름
            0.75, color, line)

    end_time = time.time()  #시간
    process_time = end_time - start_time #프로세스 시간
    print("=== A frame took {:.3f} seconds".format(process_time)) #프로세스 시간 보여주기.
    cv2.imshow("Recognition", image)
    
# 아는 얼굴 불러오고 임베딩
data = pickle.loads(open(encoding_file, "rb").read())

# 인풋이미지 불러오기
image = cv2.imread(image_file)
detectAndDisplay(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
