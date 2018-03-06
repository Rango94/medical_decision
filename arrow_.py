import cv2
import image_fuction_cache
import numpy as np
for i in range(25):
    print(i+1)
    name='test ('+str(i+1)+').jpg'
    img=cv2.imread(name)
    h,w=img.shape[:2]
    img=cv2.resize(img,(1000,int(h/w*1000)))
    blured = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh=cv2.threshold(blured,190, 255, cv2.THRESH_BINARY)
    thresh=image_fuction_cache.fillblank(thresh)
    _,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    pointlist=[]
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if (w>16 or h>16) and w!=1000 and y>100 and y<680 and w>5 and h>5:
            tmp=image_fuction_cache.czty(thresh[y: y + h, x: x + w].copy())
            n=0
            mark=1
            for i in range(tmp[1][0].size-1):
                if tmp[1][0][i]>tmp[1][0][i+1] and mark==1:
                    n+=1
                    mark=0
                if tmp[1][0][i]<tmp[1][0][i+1] and mark==0:
                    n+=1
                    mark=1
            mg = int(tmp[1].size / 3)
            vararr = []
            for k in range(3):
                aiy = np.var(tmp[1][0][mg * k:mg * (k + 1):1])
                vararr.append(aiy)
            if ((vararr[1]<=vararr[2] and vararr[1]<10 ) or (vararr[0]>10*vararr[1])) and n<50:
                if w<h*1.5:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                else:
                    tmp1 = 0
                    count=0
                    for i in tmp[1][0][mg :mg * 2:1]:
                        if i >= tmp1:
                            count+=1
                            tmp1 = i
                    if count>tmp[1][0][mg :mg * 2:1].size*0.7 and np.mean(tmp[1][0][mg :mg * 2:1])<5:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

