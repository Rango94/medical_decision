import cv2
import image_fuction_cache
# import datatosql
import os
import gensim
import numpy as np
# datatosql.refreshtable('predata')

# path=image_fuction_cache.readpath('E:/1111')

k=0
for i in range(20):
    k+=1
    print(k)
    img=cv2.imread('tmpimg.jpg')
    setw = 1200
    h, w = img.shape[:2]
    img = cv2.resize(img, (setw, int(h / w * setw)))
    h, w = img.shape[:2]
    blured = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(blured, 200, 255, cv2.THRESH_BINARY)
    thresh_cp=thresh.copy()
    allarr=image_fuction_cache.dealimg(thresh, setw)
    if allarr==0:
        continue
    newarr=image_fuction_cache.makenewarr(allarr, thresh_cp, 0)
    #新表结构：节点页码 	节点编号    	节点内容    	父节点		子节点		节点类型	     来源   转移条件  备注
    # 表结构：【【坐标,【箭头所在位置坐标（如果是箭头）,左右竖线定位位置（如果是文本）】】，节点编号（包括符号和文本），节点类型，【竖线范围，【竖线起始坐标】，指向数（文本和竖线为null）】，【【前驱指向节点】，【前驱文本节点】】，【【后继指向节点】，【后继文本节点】】】
    for i in allarr:
        print(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if i[2]==0 and i[0][0]!=[-1,-1,-1,-1]:
            img = cv2.putText(img, str(i[1]), (i[0][0][0],i[0][0][1]), font, 1.2, (0, 255, 0), 2)
    for i in newarr:
        print(i)
    cv2.imshow("xxxx.jpg", thresh)
    cv2.imshow("xxxy.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

