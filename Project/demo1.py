import time
import cv2
from cvzone.HandTrackingModule import HandDetector

class Button:
    def __init__(self, pos, text, size=None):
        if size is None:
            size = [85, 85]
        self.pos = pos
        self.text = text
        self.size = size

def showOne(button, img, color=None):
    if color is None:
        color = [255, 0, 255]
    x, y = button.pos
    w, h = button.size
    cv2.rectangle(img, button.pos, (x + w, y + h), color, cv2.FILLED)
    cv2.putText(img, button.text, (button.pos[0] + 20, button.pos[1] + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

def drawAll(img, buttonList):
    for button in buttonList:
        showOne(button, img)


detector = HandDetector(detectionCon=0.8)   # 置信度设为0.8
keys = [['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/']]
finalText = ''
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
buttonList = []
for i in range(len(keys)):
    for j in range(len(keys[i])):
        buttonList.append(Button([100 * (j + 1), 100 * (i + 1)], keys[i][j]))

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, boxInfo = detector.findPosition(img)
    drawAll(img, buttonList)

    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                l, _, _ = detector.findDistance(8, 12, img, draw=False)
                # print(l)
                if l < 39:
                    showOne(button, img, [0, 255, 0])
                    finalText += button.text
                    time.sleep(0.15)
                else:
                    showOne(button, img, [175, 0, 175])

    cv2.rectangle(img, (100, 400), (800, 485), (150, 0, 150), cv2.FILLED)
    cv2.putText(img, finalText, (100, 460), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255),
                4)
    if len(finalText) > 15:
        finalText = ''

    cv2.imshow("Image", img)
    cv2.waitKey(1)
