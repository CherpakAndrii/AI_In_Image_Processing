from Lab4_utils import *
import pickle
with open("encodings.pickle", "rb") as f:
    name_encodings_dict = pickle.load(f)

imagePoroshenko = cv2.imread("examples/1.jpg")
imageArnold = cv2.imread("examples/2.jpg")
imageMultiple = cv2.imread("examples/3.jpg")
imagePoroh = cv2.imread("examples/5.jpg")
imageMe = cv2.imread("examples/4.jpg")

images = [imagePoroshenko, imageArnold, imageMultiple, imageMe, imagePoroh]

for i in range(len(images)):
    encodings = face_encodings(images[i])
    names = []
    for encoding in encodings:
        counts = {}
        for (name, encodings) in name_encodings_dict.items():
            counts[name] = nb_of_matches(encodings, encoding)
        if all(count == 0 for count in counts.values()):
            name = "Unknown"
        else:
            name = max(counts, key=counts.get)
        names.append(name)
    for rect, name in zip(face_rects(images[i]), names):
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        cv2.rectangle(images[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(images[i], name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow("image", images[i])
    cv2.waitKey(0)
