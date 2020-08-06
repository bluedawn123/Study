import cv2
import face_recognition
import pickle

dataset_paths = ['./2FinalProject/data/son/', './2FinalProject/data/kang/', './2FinalProject/data/tedy/', './2FinalProject/data/beckham/']
names = ['Son', 'Kang', 'Tedy', 'Beckham']  
number_images = 10
image_type = '.jpg'
encoding_file = 'encodings.pickle'
# Either cnn  or hog. The CNN method is more accurate but slower. HOG is faster but less accurate.
model_method = 'cnn'

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, dataset_path) in enumerate(dataset_paths):
    # extract the person name from names
    name = names[i]                                                  #i를 이용해 names[0]이 손이므로 name으로 지정. 그것을 또 아래의 for문으로.                                      
    for idx in range(number_images):
        file_name = dataset_path + str(idx+1) + image_type           #파일 이름을 저장 방식

        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(file_name)                 #cv2.imread를 사용해서 하나씩 읽는다. 
                                                      
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #opencv는 bgr로 되어 있는데 rgb로 변형해야 한다. 

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        
        #얼굴 탐색을 위한 BOX생성
        boxes = face_recognition.face_locations(rgb,    
            model=model_method)

        # compute the facial embedding for the face
        # rgb와 box가 생성되었으므로, face_recognition.face_encodings를 사용해 인코딩한다.
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        # 위의 인코딩 한 것을을 다시 for문으로.
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            print(file_name, name, encoding)
            knownEncodings.append(encoding)
            knownNames.append(name)
        
# Save the facial encodings + names to disk
data = {"encodings": knownEncodings, "names": knownNames} #배열값
f = open(encoding_file, "wb") # open을 사용해 fileopen. 
f.write(pickle.dumps(data))   # write로 pickle에 dump를 사용해 데이터를 넣어준다. 
f.close()
