import cv2


if __name__ == '__main__':
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cv2.startWindowThread()
    cap = cv2.VideoCapture("zaluzhnyi.MP4")
    while True:
        ret, frame = cap.read()
        if not ret or cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame = cv2.resize(frame, (800, 560))
        gray_filter = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        face_rects = face_classifier.detectMultiScale(gray_filter, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.imshow("Zaluzhnyi", frame)
    cap.release()
    cv2.destroyAllWindows()
