import cv2
import numpy as np

GRID_SIZE = (16*4,16*4)

img = np.random.randint(2, size=GRID_SIZE, dtype=np.uint8) * 255

cv2.imwrite("code_{}x{}.png".format(GRID_SIZE[0], GRID_SIZE[1]), img)
