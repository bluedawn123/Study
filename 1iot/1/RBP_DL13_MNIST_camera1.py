import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tensorflow as tf
import time

model = tf.keras.models.load_model('D:/Study/2FinalProject/data/digits_model.h5')
default_src = 0
title_name = '다크호스조. 웹캠을 통한 손글씨 인식'
SZ = 28
frame_width = 350
frame_height = 350
cap = cv2.VideoCapture(default_src)  #카메라 불러오기


def start():
    # 비디오 스트림 초기화와 카메라 센서 허용
    global cap
    default_src = int(srcSpin.get())         #스피너에서 값을 받아온다. 
    #cap = VideoStream(usePiCamera=1).start()  #pi카메라 실행경우 사용
    cap = cv2.VideoCapture(default_src)      #스피너에서 받아온 값을 넣는다. 
    #time.sleep(2.0)                         #웜업을 위해 2초간 쉬는것. 필요없다. 
    detectAndDisplay()

def detectAndDisplay():  #인식 및 보여주기
    _, frame = cap.read()  #cap에서 읽어온 것을 frame에 넣어준다. 
    width = frame.shape[1]   
    height = frame.shape[0]
    width_ratio = frame_width / width
    height_ratio = frame_height / height

    frame = cv2.resize(frame, (frame_width, frame_height))  #윤곽선.
    # hsv transform - value = gray image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)            
    hue, saturation, value = cv2.split(hsv)
    # kernel to use for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # applying topHat operations
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    # applying blackHat operations
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
    # add and subtract between morphological operations
    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)

    # applying gaussian blur on subtract image
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)
    # thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    #cv2.imshow('thresh', thresh)
    
    # cv2.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3
    cv2MajorVersion = cv2.__version__.split(".")[0] 

    #오픈 cv버젼이 바뀌면서 버젼 문제에 따른 문제를 해결하기 위해서 나눠줌

    # check for contours on thresh
    if int(cv2MajorVersion) >= 4:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_digits = []
    positions = []
    margin = 30

    img_origin = frame.copy()

    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)

      # Ignore small sections
      if w * h < 2400: continue
      y_position = y-margin
      if(y_position < 0): y_position = 0
      x_position = x-margin
      if(x_position < 0): x_position = 0
      img_roi = thresh[y_position:y+h+margin, x_position:x+w+margin]
      num = cv2.resize(img_roi, (SZ,SZ))
      num = num.astype('float32') / 255.
      
      result = model.predict(np.array([num]))
      result_number = np.argmax(result)
      cv2.rectangle(frame, (x-margin, y-margin), (x+w+margin, y+h+margin), (0, 255, 0), 2)
      
      text = "Number is : {} ".format(result_number)
      cv2.putText(frame, text, (margin, frame_height-margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detectAndDisplay)     
        
#main
main = Tk()
main.title(title_name)
main.geometry()

#Graphics window (그냥 실행창임!)
label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)
srcLabel=Label(main, text='Video Source : ')
srcLabel.grid(row=1,column=0)
srcVal  = IntVar(value=default_src)
srcSpin = Spinbox(main, textvariable=srcVal,from_=0, to=5, increment=1, justify=RIGHT)
srcSpin.grid(row=1, column=1)

Button(main,text="Start", height=2,command=lambda:start()).grid(row=1, column=2, columnspan=2, sticky=(W, E))
imageFrame = Frame(main)
imageFrame.grid(row=2,column=0,columnspan=4)
  
#Capture video frames
lmain = Label(imageFrame)
lmain.grid(row=0, column=0)

main.mainloop()  #Starts GUI
