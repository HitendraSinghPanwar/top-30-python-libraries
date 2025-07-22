
#scipy

from scipy import linalg
import numpy as np

x = np.array([[16, 4], [100, 25]])

print("\nMatrix square root:\n", linalg.sqrtm(x))
print("\nMatrix exponential:\n", linalg.expm(x))
print("\nMatrix sine:\n", linalg.sinm(x))
print("\nMatrix cosine:\n", linalg.cosm(x))
print("\nMatrix tangent:\n", linalg.tanm(x))

