import cv2


if __name__ == '__main__':
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    for photo in ("poroshenko.png", "ushchenko.jpg", "homies.jpg"):
        frame = cv2.imread(photo)
        gray_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_classifier.detectMultiScale(gray_filter, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in face_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            roi_gray = gray_filter[y:y+h, x:x+w]
            roi_color = frame [y:y+h, x:x+w]
            smile = smile_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
            eye = eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=5)
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
            for (ex, ey, ew, eh) in eye:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)
        print(f"Found {len(face_rects)} faces!")
        cv2.imshow(photo, frame)
        cv2.waitKey()
        cv2.destroyAllWindows()