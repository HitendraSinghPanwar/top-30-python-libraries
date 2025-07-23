from PIL import Image
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

def visualize(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

image_path = r"C:\Users\thaku\Downloads\OIP (2).jpeg"
image = Image.open(image_path)
visualize(image)

image_np = np.array(image)

transform = A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1)
transformed = transform(image=image_np)
visualize(transformed['image'])

transform = A.RandomSnow(brightness_coeff=2.5, p=1)
transformed = transform(image=image_np)
visualize(transformed['image'])

transform = A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=1)
transformed = transform(image=image_np)
visualize(transformed['image'])

transform = A.RandomShadow(shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=1)
transformed = transform(image=image_np)
visualize(transformed['image'])
