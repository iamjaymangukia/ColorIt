import imgaug.augmenters as iaa
import cv2

img = cv2.imread("images/amg.jpg")

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1280, 720)
cv2.imshow("Image", img)
cv2.waitKey(0)