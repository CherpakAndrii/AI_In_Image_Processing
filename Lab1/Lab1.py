import imutils as imutils
import numpy as np
import cv2


def show(label: str, image_to_show: np.ndarray) -> None:
    cv2.imshow(label, image_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_m(images_to_show: list[tuple[str, np.ndarray]]) -> None:
    for label, picture in images_to_show:
        cv2.imshow(label, picture)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('sample.bmp')
    img_gray = cv2.imread('sample.bmp', 0)

    show_m([('standard', img), ('grey', img_gray)])

    cv2.imwrite('grey_sample.jpg ', img_gray)

    (h, w, d) = img.shape
    print(f"width={w}, height={h}, depth={d}")

    (B, G, R) = img[50, 50]
    print(f"R={R}, G={G}, B={B}")

    central_part = img[142:284, 214:426]
    show("central part", central_part)

    new_height = 600
    ratio = w/h
    new_width = int(new_height*ratio)

    resized_img = cv2.resize(img, (new_width, new_height))
    show('resized', resized_img)

    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, -45, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    show("OpenCV Rotation", rotated)

    rotated2 = imutils.rotate(img, 45)
    show("Imutils Rotation", rotated2)

    blurred5 = cv2.GaussianBlur(img, (5, 5), 0)
    blurred11 = cv2.GaussianBlur(img, (11, 11), 0)
    blurred15 = cv2.GaussianBlur(img, (15, 15), 0)
    blurred21 = cv2.GaussianBlur(img, (21, 21), 0)
    blurred51 = cv2.GaussianBlur(img, (51, 51), 0)
    blurred91 = cv2.GaussianBlur(img, (91, 91), 0)
    show_m([("Blurred (5x5)", blurred5), ("Blurred (11x11)", blurred11), ("Blurred (15x15)", blurred15),
            ("Blurred (21x21)", blurred21), ("Blurred (51x51)", blurred51), ("Blurred (91x91)", blurred91)])

    #suming = np.dstack((np.hstack((blurred5, blurred11)), np.hstack((blurred15, blurred21)), np.hstack((blurred51, blurred91))))
    #show("suming", suming)

    img_for_drawing = img.copy()
    cv2.rectangle(img_for_drawing, (120, 300), (200, 370), (0, 0, 255), 2)
    show('rectangle', img_for_drawing)

    img_for_drawing = img.copy()
    cv2.line(img_for_drawing, (400, 120), (530, 160), (0, 0, 255), 2)
    cv2.line(img_for_drawing, (520, 140), (530, 160), (0, 0, 255), 2)
    cv2.line(img_for_drawing, (510, 165), (530, 160), (0, 0, 255), 2)
    show('line', img_for_drawing)

    cv2.circle(img_for_drawing, (550, 170), 20, (0, 0, 255), 2)
    show('circle', img_for_drawing)

    img_for_drawing = img.copy()
    star_points = np.array ([[290, 155], [350, 340], [195, 225], [380, 225], [230, 340]])
    cv2.polylines(img_for_drawing, np.int32([star_points]), 1, (255, 0, 0), 5)
    show('star', img_for_drawing)

    img_for_drawing = img.copy()
    font1 = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    font2 = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_for_drawing, 'Postavte zalik', (30, 150), font1, 3, (255, 255, 255), 2, cv2.LINE_4)
    cv2.putText(img_for_drawing, "bud' laska..", (5, 320), font2, 4, (255, 255, 255), 4, cv2.LINE_4)
    show("zalik", img_for_drawing)
    cv2.imwrite('zalik.jpg ', img_for_drawing)
