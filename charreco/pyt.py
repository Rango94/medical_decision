import pytesseract
from PIL import Image

import cv2
img=cv2.imread('test (2).jpg')
img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
code = pytesseract.image_to_string(img)
print (code)