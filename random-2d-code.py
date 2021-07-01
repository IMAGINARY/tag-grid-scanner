import cv2
import numpy as np

GRID_SHAPE = (16 * 4, 16 * 4)

img = np.random.randint(2, size=GRID_SHAPE, dtype=np.uint8) * 255

cv2.imwrite("code_{}x{}.png".format(GRID_SHAPE[0], GRID_SHAPE[1]), img)
