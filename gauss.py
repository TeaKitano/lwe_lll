from math import floor
import numpy as np

def gaussian_reduction(v1, v2):
    while True:
        if np.linalg.norm(v2) < np.linalg.norm(v1):
            v1, v2 = v2, v1
        m = floor(np.dot(v1, v2) / np.dot(v1, v1))
        if m == 0:
            break
        v2 = v2 - m * v1
    return v1, v2

if __name__ == "__main__":
    v1 = np.array([2, 3])
    v2 = np.array([1, 4])
    print(gaussian_reduction(v1, v2))