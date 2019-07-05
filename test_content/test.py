import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

x = cv2.imread('rit_snow_0_label.png')
x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
plt.imshow(x)
plt.show()