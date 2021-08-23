import cv2
from cvzone.HandTrackingModule import  HandDetector

class Button:
    def __init__(self, pos, text, size=None):
        if size is None:
            size = [100, 100]
        self.pos = pos
        self.text = text
        self.size = size


detector = HandDetector(detectionCon=0.8)   # 置信度设为0.8
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, boxInfo = detector.findPosition(img)
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 255), cv2.FILLED)
    cv2.putText(img, 'Q', (125, 180), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
