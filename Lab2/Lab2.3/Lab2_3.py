import cv2
import numpy as np
from get_video import download

if __name__ == '__main__':
    # download("https://www.youtube.com/watch?v=_mPelM7Yaq8")
    # download("https://www.youtube.com/watch?v=UWiinhrRcKA")
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cv2.startWindowThread()
    cap = cv2.VideoCapture("parad.mp4")
    while True:
        ret, frame = cap.read()
        if not ret or cv2.waitKey(1) & 0XFF == ord("q"):
            break
        frame = cv2.resize(frame, (800, 560))
        gray_filter = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        face_rects = face_classifier.detectMultiScale(gray_filter, scaleFactor=1.1, minNeighbors=5)
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        for (xa, ya, xb, yb) in boxes:
            cv2.rectangle(frame, (xa, ya), (xb, yb), (0, 255, 255), 1)
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.imshow("Video", frame)

    cap.release()
    cv2.destroyAllWindows()
