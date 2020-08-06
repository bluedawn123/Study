from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt   

from keras.preprocessing.image import load_img
import os
path = 'D:/Study/data/dog_cat/'
os.chdir(path)

img_dog = load_img(path + 'dog.jpg', target_size=(224, 224))
img_cat = load_img(path + 'cat.jpg', target_size=(224, 224))
img_suit = load_img(path + 'suit.jpg', target_size=(224, 224))
img_yangpa = load_img(path + 'yangpa.jpg', target_size=(224, 224))

plt.imshow(img_yangpa)


#이미지를 넘파이 화
from keras.preprocessing.image import img_to_array

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_yangpa = img_to_array(img_yangpa)

print(arr_dog)
print(type(arr_dog)) 
print(arr_dog.shape)

#RGB -> BGR
from keras.applications.vgg16 import preprocess_input

arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_yangpa = preprocess_input(arr_yangpa)
arr_suit = preprocess_input(arr_suit)

print(arr_dog)
print(arr_dog.shape)


#이미지 데이터를 하나로 합친다. 
import numpy as np

#????????????
arr_input = np.stack([arr_dog, arr_cat, arr_suit, arr_yangpa])

print(arr_input)
model = VGG16()
probs = model.predict(arr_input)

print(probs)
print("probs.shape : ", probs.shape)


#이미지결과
from keras.applications.vgg16 import decode_predictions

results = decode_predictions(probs)


print("=----------------")
print(results[0])
print("=----------------")
print(results[1])
print("=----------------")
print(results[2])
print("=----------------")
print(results[3])
