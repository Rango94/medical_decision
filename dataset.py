import cv2
import image_fuction_cache
file=open('dataset.txt','w')
for i in range(25):
    name='test ('+str(i+1)+').jpg'
    img=cv2.imread(name)
    hs, ws = img.shape[:2]
    img = cv2.resize(img, (1000, int(hs / ws * 1000)))
    blured = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(blured, 190, 255, cv2.THRESH_BINARY)
    thresh = image_fuction_cache.fillblank(thresh)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pointlist = []
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if (w > 16 or h > 16) and w != 1000 and y > 100 and y < 680 and w > 5 and h > 5:

            tmpimg=cv2.resize(thresh[y: y + h, x: x + w],(200,200))
            cv2.imshow('show',tmpimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            mark=int(input("mark="))
            tmp = image_fuction_cache.czty(tmpimg)[1][0]
            for j in tmp:
                file.write(str(j)+' ')
            file.write(str(mark)+'\n')
file.close()


