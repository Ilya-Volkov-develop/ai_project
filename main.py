import cv2
import numpy as np


def nothing(*arf):
    pass


if __name__ == '__main__':
    print('')
    # глава 1
    # img = cv2.imread('images/stud.jpg')
    # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.Canny(img, 90, 90)
    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # cv2.imshow('Result', img)
    # cv2.waitKey(5000)
    # (1224, 404, 3)

    # глава 2
    # photo = np.zeros((450, 450, 3), dtype='uint8')
    # photo[100:150, 200:280] = 120, 100, 5
    # cv2.rectangle(photo, (0, 0), (100, 100), (120, 100, 5), thickness=3)
    # cv2.line(photo, (0, photo.shape[0] // 2), (photo.shape[1], photo.shape[0] // 2), (120, 100, 5), thickness=3)
    # cv2.circle(photo, (photo.shape[1] // 2, photo.shape[0] // 2), 100, (120, 100, 5), thickness=3)
    # cv2.putText(photo, 'hello', (150, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (120, 100, 5), 1)
    # cv2.imshow('images/stud.jpg', photo)
    # cv2.waitKey(2000)

    # глава 3
    # cap = cv2.VideoCapture(0)
    # color_spaces = ('RGB', 'GRAY', 'HSV', 'LAB', 'XYZ', 'YUV')
    # cap.set(3, 500)
    # cap.set(4, 300)
    #
    # while True:
    #     success, img = cap.read()
    #     color_image = {color: cv2.cvtColor(img, getattr(cv2, 'COLOR_BGR2' + color)) for color in color_spaces}
    #     for color in color_image:
    #         cv2.imshow(color, color_image[color])
    #     cv2.imshow('Camera', img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # глава 4

    # cap = cv2.VideoCapture(0)
    # while True:
    #     success, img = cap.read()
    #     low_blue = np.array((90, 70, 70), np.uint8)
    #     high_blue = np.array((190, 170, 170), np.uint8)
    #     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     mask_blue = cv2.inRange(img_hsv, low_blue, high_blue)
    #     result = cv2.bitwise_and(img_hsv, img_hsv, mask=mask_blue)
    #     result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    #     cv2.imshow('Camera', mask_blue)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cv2.namedWindow("result")
    # cv2.namedWindow("settings")
    # cap = cv2.VideoCapture(0)
    #
    # cv2.createTrackbar('red1', 'settings', 0, 255, nothing)
    # cv2.createTrackbar('green1', 'settings', 0, 255, nothing)
    # cv2.createTrackbar('blue1', 'settings', 0, 255, nothing)
    # cv2.createTrackbar('red2', 'settings', 0, 255, nothing)
    # cv2.createTrackbar('green2', 'settings', 0, 255, nothing)
    # cv2.createTrackbar('blue2', 'settings', 0, 255, nothing)
    # crange = [0, 0, 0, 0, 0, 0]
    #
    # try:
    #     while True:
    #         flag, img = cap.read()
    #         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #         red1 = cv2.getTrackbarPos('red1', 'settings')
    #         green1 = cv2.getTrackbarPos('green1', 'settings')
    #         blue1 = cv2.getTrackbarPos('blue1', 'settings')
    #         red2 = cv2.getTrackbarPos('red2', 'settings')
    #         green2 = cv2.getTrackbarPos('green2', 'settings')
    #         blue2 = cv2.getTrackbarPos('blue2', 'settings')
    #
    #         h_min = np.array((blue1, green1, red1), np.uint8)
    #         h_max = np.array((blue2, green2, red2), np.uint8)
    #
    #         thresh = cv2.inRange(hsv, h_min, h_max)
    #         result = cv2.bitwise_and(img, img, mask=thresh)
    #
    #         cv2.imshow('result', result)
    #         if cv2.waitKey(5) == 27:
    #             break
    # except KeyboardInterrupt:
    #     print('exit')
    #     cv2.destroyAllWindows()

    # глава 5
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, img = cap.read()
    #     cv2.imshow('Result', img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # img = cv2.imread('images/face.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # faces = cv2.CascadeClassifier('face.xml')
    #
    # results = faces.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=1)
    #
    # for (x, y, w, h) in results:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
    #
    #     cv2.imshow('Result', img)
    #     cv2.waitKey(0)
