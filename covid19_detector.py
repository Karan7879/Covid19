
#COVID19 DETECTOR
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

model = load_model('corona1.h5')

img_dim = 96

class_labels = ['COVID-19','NORMAL','Viral Pneumonia']
# test_img = cv2.imread('covid.png')
test_img = cv2.imread('NORMAL.png')
# test_img = cv2.imread('Viral_Pneumonia.png')
# test_img = cv2.cvtColor(test_img,cv2.BGR2GRAY)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY )
test_img = cv2.resize(test_img, (img_dim,img_dim))
img = image.img_to_array(test_img)
img = np.expand_dims(img, axis=0)
img = img.astype('float32')/255
pred = np.argmax(model.predict(img))

color = (255,0,255)
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(test_img, class_labels[pred], (50,50), font, 1.0, color, 2)
print(class_labels[pred])

cv2.imshow('image', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()