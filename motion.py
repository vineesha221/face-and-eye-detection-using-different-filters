import cv2
import numpy as np

img = cv2.imread(r'C:\Users\vineesha thoutam\Downloads\rdj.jpg')
cv2.imshow('Original', img)

size = 15

# generating the kernel
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

# applying the kernel to the input image
output = cv2.filter2D(img, -1, kernel_motion_blur)

cv2.imshow('Motion Blur', output)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
faces = face_cascade.detectMultiScale(gray_img, 1.3, 1)
for (x,y,w,h) in faces:
    cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray_img[y : y+h, x : x+w]
    roi_color = output[y : y+h, x : x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
cv2.imshow('image', output)
cv2.waitKey(0)