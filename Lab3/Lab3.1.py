import numpy as np
import cv2


def show(label: str, image_to_show: np.ndarray) -> None:
    cv2.imshow(label, image_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demonstrate_staps():
    img = cv2.imread("road2.jpg")
    show("original image", img)

    # конвертуємо зображення у чорнобіле
    grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show("grayscale image", grayScale)

    # розмиваємо зображення для того, аби прибрати шуми
    kernel_size = 5
    blur = cv2.GaussianBlur(grayScale, (kernel_size, kernel_size), 0)
    show("blur image", blur)

    # за допомогою алгоритму Кенні знаходимо межі об'єктів
    low_t = 50
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)
    show("edges image", edges)

    # Оскільки нас цікавлять лише об'єкти у межах певної області, накладемо на зображення маску
    vertices = np.array(
        [[(0, img.shape[0]), (450, 310), (490, 310), (img.shape[1], img.shape[0])]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    show("mask image", mask)
    masked_edges = cv2.bitwise_and(edges, mask)
    show("masked_edges image", masked_edges)


# Метод для малювання ліній дорожньої розмітки на зображенні
def draw_lines(frame, lines, color=[0, 0, 255], thickness=10):
    x_bottom_pos = []
    x_upper_pos = []
    x_bottom_neg = []
    x_upper_neg = []
    y_bottom = 540
    y_upper = 315
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = ((y2 - y1) / (x2 - x1))
            b = y1 - slope * x1
            if slope > 0.5 and slope < 0.8:
                x_bottom_pos.append((y_bottom - b) / slope)
                x_upper_pos.append((y_upper - b) / slope)
            elif slope < -0.5 and slope > -0.8:
                x_bottom_neg.append((y_bottom - b) / slope)
                x_upper_neg.append((y_upper - b) / slope)
    if len(x_bottom_pos) > 0 and len(x_bottom_neg) > 0:
        lines_mean = np.array(
            [[int(np.mean(x_bottom_pos)), int(np.mean(y_bottom)), int(np.mean(x_upper_pos)), int(np.mean(y_upper))],
             [int(np.mean(x_bottom_neg)), int(np.mean(y_bottom)), int(np.mean(x_upper_neg)), int(np.mean(y_upper))]])
        for i in range(len(lines_mean)):
            cv2.line(frame, (lines_mean[i, 0], lines_mean[i, 1]),
                     (lines_mean[i, 2], lines_mean[i, 3]), color, thickness)


# Метод для обробки зображення - пошуку на ньому дорожньої розмітки
def process_image(frame):
    vertices = np.array(
        [[(0, frame.shape[0]),
          (450, 310),
          (490, 310),
          (frame.shape[1], frame.shape[0])]], dtype=np.int32)
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayScale, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(
        masked_edges, 3, np.pi / 180, 15, np.array([]),
        minLineLength=100,
        maxLineGap=70)
    draw_lines(frame, lines)


# метод для покадрової обробки та відображення відео
def process_video():
    cv2.startWindowThread()
    video_capture = cv2.VideoCapture("road.mp4")
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            process_image(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demonstrate_staps()
    process_video()