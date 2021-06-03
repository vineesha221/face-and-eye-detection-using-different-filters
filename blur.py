import cv2
import numpy as np
img = cv2.imread(r'C:\Users\vineesha thoutam\Downloads\ladka.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

rows,cols = img.shape[:2]



cv2.imshow('Original',img)
#blur
kernel_3x3 = np.ones((3,3),np.float32) / 9.0
output1=cv2.filter2D(img,-1,kernel_3x3) 
cv2.imshow('Identity filter',output1)

#larger blur
kernel_5x5 = np.ones((5,5),np.float32) / 25.0
output2=cv2.filter2D(img,-1,kernel_5x5)
cv2.imshow('Identity filter1',output2)

#motion blur
size=15
kernel_motion_blur = np.zeros((size,size))
kernel_motion_blur[int((size-1)/2),:] = np.ones(size)
kernel_motion_blur = kernel_motion_blur /  size
output3=cv2.filter2D(img,-1,kernel_motion_blur)
cv2.imshow('Motion blur',output3)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
faces = face_cascade.detectMultiScale(gray_img, 1.3, 1)
for (x,y,w,h) in faces:
    cv2.rectangle(output3, (x, y), (x+w, y+h), (0, 0, 255), 2)
    roi_gray = gray_img[y : y+h, x : x+w]
    roi_color = output3[y : y+h, x : x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
cv2.imshow('image', output3)
cv2.waitKey(0)