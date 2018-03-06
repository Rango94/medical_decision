import cv2
import image_fuction_cache
path=image_fuction_cache.readfile('G:/nccn')
for k in path:
    name='traindata1_back.txt'
    print(k)
    name1=k
    img=cv2.imread(name1)
    setw = 1200
    h, w = img.shape[:2]
    img = cv2.resize(img, (setw, int(h / w * setw)))
    h, w = img.shape[:2]
    blured = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(blured, 200, 255, cv2.THRESH_BINARY)
    thresh_cp=thresh.copy()
    image_fuction_cache.extractword(thresh, setw, name)


