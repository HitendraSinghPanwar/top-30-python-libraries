
#matplotlib

import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\thaku\Downloads\OIP (1).jpeg")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Sample Image")
plt.axis("on")
plt.show()



import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax2 = fig.add_axes([0.6, 0.6, 0.25, 0.25])

ax1.plot([2, 3, 4, 5, 5, 6, 6],
         [1, 8, 4, 9, 3, 2, 8])
ax2.plot([1, 2, 3, 4, 5],
         [2, 3, 4, 5, 6])

plt.show()