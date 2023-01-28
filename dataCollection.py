#import libraries
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

#Webcam
cap = cv2.VideoCapture(0)
#Number of hands detected
detector = HandDetector(maxHands=1)
#space for hand to fit in frame
offset= 20
imgSize =300

# saved images location
folder = "images/Nice"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        #window with white backgownd is in the middle
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        #Window size around hand to prevent crash
        if aspectRatio >1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal) / 2)
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imgSize / h
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        #ouput of two small windows
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow('Image', img)
    #when S key is pressed image will save
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg',imgWhite)
        print(counter)