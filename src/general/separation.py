import cv2
# import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('imgs/cats_on_couch.jpg')

low_pink, high_pink = (130, 0, 0), (190, 255, 255)
mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), low_pink, high_pink)
# inverse mask
mask = 255 - mask
img2 = cv2.bitwise_and(img, img, mask=mask)
img2 = cv2.GaussianBlur(img2, (5, 5), 0)

kernel = np.ones((5, 5), np.uint8)
img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)

ret, mask2 = cv2.threshold(
    cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
    30, 256, cv2.THRESH_BINARY)
img3 = cv2.bitwise_and(img, img, mask=mask2)

output_image = np.concatenate(
    (img, img2, img3), axis=1)

cv2.imshow("threshold", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
