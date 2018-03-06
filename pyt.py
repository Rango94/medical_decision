import pytesseract
from PIL import Image
import cv2

def imgtostr(img,flag=True):
    # cv2.imshow('jdskfjls',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    img = cv2.dilate(img, kernel2)
    h,w=img.shape
    if w<=200:
        img = cv2.resize(img, (200, int(h / w * 200)))
    img = Image.fromarray(img)
    string=pytesseract.image_to_string(img)
    string=string.replace("\n"," ").replace('\t',' ')
    ' '.join(string.split())
    if flag==True:
        return string
    else:
        string=string.split('.')
        if len(string)>=2:
            i=1
            while i<len(string):
                flag=0
                for each in string[i]:
                    if each.isalpha():
                        flag=1
                        if each.islower():
                            string[i-1]=string[i-1]+" "+string[i]
                            del string[i]
                        else:
                            i+=1
                        break
                if flag==0:
                    i+=1
        return string


