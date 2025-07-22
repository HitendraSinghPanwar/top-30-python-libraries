#openCV

import cv2

img = cv2.imread(r"C:\Users\thaku\Downloads\OIP (1).jpeg", cv2.IMREAD_COLOR)

image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

image1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )

cv2.imshow("Original", image)
cv2.imshow("Grayscale", image1)

cv2.waitKey(0)
cv2.destroyAllWindows()