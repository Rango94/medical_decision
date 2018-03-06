import numpy as np
import cv2
import math
import pyt
import gensim
import nltk.corpus
from sklearn.preprocessing import StandardScaler
from sklearn import svm

#矩形逼近
def drawwall(img,img_bak):
    '''
    函数功能：形态学变化后使用矩形框定位文本节点位置。
    由于形态学变化后的节点区域与实际区域有细微差别，直接使用自带函数框选会出现错位的情况
    此函数为了精确的框选文本节点。
    :param img:形态学变化后的图片
    :param img_bak: 原图片
    :return: 标记框选后的原图片，矩形框信息
    详见文档示例1
    '''
    def optimize(img, shape, kr):
        hs, ws = img.shape
        x, y, w, h = shape
        xf, yf, wf, hf = 0, 0, 0, 0
        up = hs
        for i in range(max(x - kr, 0), min(x + kr, ws)):
            tmp = 0
            for j in range(max(y - kr, 0), min(y + h + kr, hs)):
                if img[j][i] == 0:
                    tmp += 1
            if tmp <= up:
                xf = i
                up = tmp
            if tmp > up:
                break
        up = hs
        for i in range(min(x + w + kr, ws) - 1, max(x + w - kr, 0) - 1, -1):
            tmp = 0
            for j in range(max(y - kr, 0), min(y + h + kr, hs)):
                if img[j][i] == 0:
                    tmp += 1
            if tmp <= up:
                wf = i - xf
                up = tmp
            if tmp > up:
                break
        right = ws
        for j in range(max(y - kr, 0), min(y + kr, hs)):
            tmp = 0
            for i in range(max(x - kr, 0), min(w + x + kr, ws)):
                if img[j][i] == 0:
                    tmp += 1
            if tmp <= right:
                yf = j
                right = tmp
            if tmp > right:
                break
        right = ws
        # for j in range(max(y - kr+h, 0), min(y + kr+h, hs)):
        for j in range(min(y + kr + h, hs) - 1, max(y - kr + h, 0) - 1, -1):
            tmp = 0
            for i in range(max(x - kr, 0), min(w + x + kr, ws)):
                if img[j][i] == 0:
                    tmp += 1
            if tmp <= right:
                hf = j - yf
                right = tmp
            if tmp > right:
                break
        out = xf, yf, wf, hf
        return out
    pointset=[]
    if len(img.shape)==2:
        hs,ws=img.shape
        for i in range(hs):
            for j in range(ws):
                if i==0 or i==hs-1 or j==0 or j==ws-1:
                    img[i][j]=255
    elif len(img.shape)==3:
        hs,ws,ds=img.shape
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    pointlist = []
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if w < ws or h < hs:
            pointlist.append([x, y, w, h])
    i = 0
    lenth = len(pointlist)
    while i < lenth-1:
        j = 0
        while j < lenth:
            if j == i and j != lenth - 1:
                j += 1
            if pointlist[i][0] >= pointlist[j][0] and pointlist[i][1] >= pointlist[j][1] and pointlist[i][0] + \
                    pointlist[i][2] <= pointlist[j][2] + pointlist[j][0] and pointlist[i][3] <= pointlist[j][3] - (
                pointlist[i][1] - pointlist[j][1]):
                del pointlist[i]
                lenth = len(pointlist)
                break
            else:
                j += 1
        if j == lenth:
            i += 1
    if len(img.shape)==2:
        for i in pointlist:
            x, y, w, h = i
            x,y,w,h,=optimize(img_bak,i,5)
            pointset.append([x,y,w,h])
            cv2.rectangle(img_bak, (x, y), (x + w, y + h), 0, 1)
    elif len(img.shape)==3:
        for i in pointlist:
            x, y, w, h = i
            cv2.rectangle(img_bak, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return img_bak, pointset

#形态学变换
def morphological(img, p, p1, s, s1):
    '''
    函数功能：对目标图片做形态学处理，先腐蚀后膨胀。
    :param img: 目标图片
    :param p: 水平腐蚀系数
    :param p1: 垂直腐蚀系数
    :param s: 水平膨胀系数
    :param s1: 垂直膨胀系数
    :return: 处理后的图片，矩形框信息
    详见文档示例2
    '''
    img_bak=img.copy()
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (p, p1))
    eroded = cv2.erode(img, kernel1)
    eroded=fillblank(eroded)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (s, s1))
    dilated = cv2.dilate(eroded, kernel2)
    dilated,pointset=drawwall(dilated,img_bak)
    return dilated,pointset
#去掉边缘的连通像素
def cutedgeipx(img):
    '''
    未使用
    :param img:
    :return:
    '''
    h_img, w_img = img.shape
    for i in range(h_img):
        for j in range(w_img):
            if i==0 or i==h_img-1 or j==w_img-1 or j==0:
                img=findblack(i,j,img)
            else:
                continue
    return img
def findblack(y,x,img):
    '''
    未使用
    :param y:
    :param x:
    :param img:
    :return:
    '''
    h,w=img.shape
    if img[y][x]==255:
        return img
    else:
        img[y][x]=255
        if img[max(0,y-1)][max(0,x-1)]==0:
            img=findblack(max(0,y-1),max(0,x-1),img)
        if img[max(0,y)][max(0,x-1)]==0:
            img=findblack(max(0,y),max(0,x-1),img)
        if img[min(h-1,y+1)][max(0,x-1)]==0:
            img=findblack(min(h-1,y+1),max(0,x-1),img)
        if img[max(0,y-1)][max(0,x)]==0:
            img = findblack(max(0, y - 1), max(0, x ), img)
        if img[min(h-1,y+1)][max(0,x)]==0:
            img = findblack(min(h-1,y+1), max(0, x ), img)
        if img[max(0,y-1)][min(w-1,x+1)]==0:
            img = findblack(max(0, y - 1), min(w-1,x+1), img)
        if img[max(0,y)][min(w-1,x+1)]==0:
            img = findblack(max(0, y ), min(w-1,x+1), img)
        if img[min(h-1,y+1)][min(w-1,x+1)]==0:
            img = findblack(min(h-1,y+1), min(w-1,x+1), img)
    return img
#填充空洞
def fillblank(img):
    '''
    函数功能：腐蚀图片之后，填充被隔绝的空洞，以便膨胀操作
    :param img: 图片
    :return: 处理后的图片
    详见文档示例3
    '''
    flooded=img.copy()
    out=cv2.copyMakeBorder(flooded,1,1,1,1,cv2.BORDER_CONSTANT,value=255)
    h,w=out.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(out, mask, (w - 1, h - 1),0, 2, 3, 8)
    for i in range(1,h-1):
        for j in range(1,w-1):
            if out[i][j]==255:
                img[i-1][j-1]=0
    return img
#找到箭头
def arrowandline(img,setw):
    '''
    未使用
    :param img:
    :param setw:
    :return:
    '''
    thresh = fillblank(img.copy())
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pointlist_arrow = []
    pointlist_line=[]
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if (w > 0.015 * setw or h > 0.015 * setw) and w != setw and y > 0.1 * setw and y < 0.68 * setw :
            if w<0.005*setw and h>w*3:
                pointlist_line.append([x,y,w,h])
                continue
            if w < 0.010 * setw or (h < 0.010 * setw and w < h):
                continue
            N = 0
            n = 0
            tmpimg = thresh[y: y + h, x: x + w]
            for i in tmpimg:
                for j in i:
                    N += 1
                    if j == 0:
                        n += 1
            tmpimg1=thresh[y+int(h/2):y+int(h/2)+1,x:x+w]
            N1=0
            n1=0
            for i in tmpimg1:
                for j in i:
                    N1+=1
                    if j==0:
                        n1+=1
            if float(n / N) < 0.28 or n1/N1>0.8:
                pointlist_arrow.append([x,y,w,h])
    return pointlist_arrow,pointlist_line

# 去掉图片中的箭头
def blankarrowandline(img,pointlist,line):
    '''
    函数功能：使待处理图片中的箭头和竖线空白
    :param img: 待处理图片
    :param pointlist: 箭头信息
    :param line: 竖线信息
    :return: 处理后的图片
    详见文档示例4
    '''
    for i in line:
        pointlist.append(i)
    for i in pointlist:
        x, y, w, h=i
        mark=0
        tmpimg = img[y: y + h, x: x + w].copy()
        img[y: y + h, x: x + w]=np.ones((h,w))*255
        mask = np.ones((h + 2, w + 2), np.uint8)*255
        mask[1:1+h,1:1+w]=tmpimg
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)!=2:
            for j in contours:
                x1, y1, w1, h1 = cv2.boundingRect(j)
                if w1<0.4*w or h1<0.4*h:
                    img[y1+y: y+y1 + h1-2, x+x1: x+x1 + w1-2]=mask[y1+1: y1 + h1-1, x1+1: x1 + w1-1]
    return img

def arrowandline_num(img,setw):
    '''
    函数功能：找到待处理图片中的箭头和竖线。
    :param img: 待处理图片
    :param setw: 图片宽度
    :return: 箭头信息，竖线信息
    '''
    img_cp=img
    thresh = fillblank(img.copy())
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pointlist_arrow = []
    pointlist_line=[]
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if (w > 0.015 * setw or h > 0.015 * setw) and w != setw and y > 0.1 * setw and y < 0.68 * setw :
            if w<0.005*setw and h>w*3:
                pointlist_line.append([x,y,w,h])
                continue
            if w < 0.010 * setw or h < 0.010 * setw and w < h:
                continue
            N = 0
            n = 0
            tmpimg = img_cp[y: y + h, x: x + w]
            for i in tmpimg:
                for j in i:
                    N += 1
                    if j == 0:
                        n += 1
            N1 = 0
            n1 = 0
            if h<(20/1200)*setw:
                tmpimg1 = img_cp[y + int(h / 2):y + int(h / 2) + 1, x:x + w]
                maxx=0
                for i in tmpimg1:
                    for j in i:
                        N1 += 1
                        if j == 0:
                            n1 += 1
                            if n1>maxx:
                                maxx=n1
                        if j==255:
                            n1=0
            if (h>=(20/1200)*setw and float(n / N) < 0.28) or (h<(20/1200)*setw and (maxx/N1)>0.8):
                my,tmpimg_cp=morphological(tmpimg.copy(), 1, 1, 3, 3)
                arrowset=[]
                for tmp in tmpimg_cp:
                    if tmp[0]>w/5:
                        tmp[0]=tmp[0]+x
                        tmp[1]=tmp[1]+y
                        arrowset.append(tmp)
                cont_arr = {0: 0}
                for i in range(int(w / 3), int(2 * w / 3)):
                    cont = 0
                    for j in range(1, h):
                        if tmpimg[j, i] != tmpimg[j - 1, i]:
                            cont += 1
                    if cont not in cont_arr:
                        cont_arr[cont] = 1
                    else:
                        cont_arr[cont] = cont_arr.get(cont) + 1
                cont_arr = sorted(cont_arr.items(), key=lambda d: d[1], reverse=True)
                branch = int(cont_arr[0][0]/2)
                long=[]
                idf=0
                for i in range(0,int(w/3)):
                    cont=0
                    for j in range(1,h-1):
                        if tmpimg[j,i]==0 and idf==0:
                            xy=[i+x,j+y]
                            idf=1
                        if tmpimg[j+1,i]!=tmpimg[j,i]:
                            long.append(cont)
                            cont=0
                        if tmpimg[j,i]==0 :
                            cont+=1
                    long.append(cont)
                if arrowset!=[]:
                    if max(long)>20:
                        pointlist_arrow.append([[[x, y, w, h],arrowset], [max(long),xy,branch]])
                    else:
                        pointlist_arrow.append([[[x, y, w, h],arrowset], [0,0, branch]])
    # if pointlist_arrow!=[]:
    #     print(pointlist_arrow)
    return pointlist_arrow,pointlist_line

#计算两个候选框之间的最短距离，k为左边的候选框
def minmargin(k,i):
    '''
    函数功能：计算两个候选框（包括文本框，箭头，竖线）之间的最短距离
    :param k: 左边的候选框
    :param i: 右边的候选框
    :return: 距离以及相关信息
    '''
    set_k = []
    set_i = []
    tmp=10000
    out=[]
    for y in range(k[1], k[3] + k[1]):
        set_k.append([k[0] + k[2], y])
    for y in range(i[1], i[1] + i[3]):
        set_i.append([i[0], y])
    for m in set_k:
        for n in set_i:
            margin=(abs(m[0] - n[0]) + abs(m[1] - n[1])) / 2
            if margin<=tmp:
                tmp=margin
                out=[tmp,[m,n]]
    return out


#计算两个文本候选框之间有没有指向关系
#未使用的方法
def searcharrow_bw_word(word1,word2,list,dis):
    zpacd=[-1,-1]
    bz=10000
    if word1[2]==0 and word2[2]==0:
        for i in range(word1[1],word2[1]):
            tmp=list[i]
            if tmp[2]==1:
                for k in tmp[0][1]:
                    left=minmargin(word1[0][0],tmp[0][0])
                    right=minmargin(k,word2[0][0])
                    margin=(left[0]+right[0])/2
                    if margin<bz :
                        zpacd[0]=k
                        zpacd[1]=margin
                        bz=margin
        if zpacd[1]<dis:
            return zpacd
    if word1[2]==2 and word2[2]==0:
        for i in range(word1[1],word2[1]):
            tmp=list[i]
            if tmp[2]==1:
                for k in tmp[0][1]:
                    left=minmargin(word1[0],tmp[0][0])
                    right=minmargin(k,word2[0][0])
                    margin=(left[0]+right[0])/2
                    if margin<bz :
                        zpacd[0]=k
                        zpacd[1]=margin
                        bz = margin
        if zpacd[1]<dis :
            return zpacd
    if word1[2]==1 and word2[2]==0:
        for k in word1[0][1]:
            right=minmargin(k,word2[0][0])
            margin=right[0]
            if margin < bz:
                zpacd[0] = k
                zpacd[1] = margin
                bz = margin
        if zpacd[1]<dis :
            return zpacd
    if word1[2]==2 and word2[2]==2:
        if word1==word2:
            return [0,0]
        for i in range(word1[1],word2[1]):
            tmp=list[i]
            if tmp[2]==1:
                for k in tmp[0][1]:
                    left=minmargin(word1[0],tmp[0][0])
                    right=minmargin(k, word2[0])
                    margin = (left[0]+right[0])/2
                    if margin < bz:
                        zpacd[0] = k
                        zpacd[1] = margin
                        bz = margin
        if zpacd[1] < dis:
           return zpacd
    if word1[2]==1 and word2[2]==2:
        for k in word1[0][1]:
            right=minmargin(k,word2[0])
            margin=right[0]
            if margin < bz:
                zpacd[0] = k
                zpacd[1] = margin
                bz = margin
        if zpacd[1] < dis:
            return zpacd
    if word1[2] == 0 and word2[2] == 2:
        for i in range(word1[1],word2[1]):
            tmp=list[i]
            if tmp[2]==1:
                for k in tmp[0][1]:
                    left=minmargin(word1[0][0],tmp[0][0])
                    right=minmargin(k,word2[0])
                    margin=(left[0]+right[0])/2
                    if margin < bz:
                        zpacd[0] = k
                        zpacd[1] = margin
                        bz = margin
        if zpacd[1] < dis:
            return zpacd
    return [-1,-1]

#找到文本左右的竖线，如果没有那么返回本身
def weather_line_nr_word(word, list, specialline):
    '''
    函数功能：判断文本框附近是否有定位竖线
    :param word: 木匾文本节点信息
    :param list: 所有节点信息
    :param specialline: 特殊竖线信息
    :return: 文本框以及与之相关的定位线信息
    '''
    out=[word,word]
    for i in range(word[1],len(list)):
        if specialline!=[] and word[1] in [x[1] for x in specialline]:
            out[1]=list[specialline[[x[1] for x in specialline].index(word[1])][0]]
            break
        tmp=list[i]
        if tmp[2] == 1 and tmp[3][0] != 0:
            if weather_point_in_area(word[0][2],[tmp[3][1][0], tmp[3][1][1], 3, tmp[3][0]]):
                out[1]=tmp
                break
        if tmp[2] == 2:
            if weather_point_in_area(word[0][2],tmp[0]):
                out[1] = tmp
                break
    for i in range(0,word[1]):
        tmp = list[i]
        if tmp[2] == 2:
            if weather_point_in_area(word[0][1], tmp[0]):
                out[0] = tmp
                break
    return out

def weather_point_in_area(point,area):
    '''
    函数功能：判断一个点是否在指定区域
    :param point: 点
    :param area: 区域
    :return: 是否
    '''
    if area[0]<=point[0] and area[1]<=point[1] and area[0]+area[2]>=point[0] and area[1]+area[3]>=point[1]:
        return True
    return False


def dealimg(thresh,setw):
    '''
    函数功能：综合处理一张图片
    :param thresh: 待处理图片（二值）
    :param setw: 图片宽度
    :return: 所有节点的相关信息
    '''
    h,w=thresh.shape
    img_baw = thresh.copy()
    adr = findend(thresh, setw)
    print(adr)
    pointset_arrow, pointset_line = arrowandline_num(thresh, setw)
    if pointset_arrow==[]:
        return 0
    print([i[0][0] for i in pointset_arrow])
    # 通过膨胀和腐蚀的方法获得未经处理的文本框
    thresh[:][:adr[0]] = np.ones((adr[0], setw)) * 255
    # print(adr[1])
    thresh[:][adr[1]:] = np.ones((h - adr[1], setw)) * 255

    cv2.imshow('sdgsgs', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    thresh = blankarrowandline(thresh,[i[0][0] for i in pointset_arrow],pointset_line)
    # cv2.imshow('dddddsfsdfsd',thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    tmpimg, pointset_word_pre =morphological(thresh, 12, 9, 11, 8)
    pointset_word = []
    # 删除不满足要求的文本框，并且找到左右的标记点
    for i in pointset_word_pre:
        x, y, w, h = i
        left_xy = [-1, -1]
        right_xy = [-1, -1]
        if w>3 and h>3:
            for xx in range(x, -1, -1):
                flag = 0
                for yy in range(y + int(h / 3), y + int(h / 1.5)):
                    if img_baw[yy, xx] == 0:
                        left_xy = [xx, yy]
                        flag = 1
                        break
                if flag == 1:
                    break
                if img_baw[y + int(h / 3), xx] == 0:
                    left_xy = [xx, y + int(h / 3)]
                    break
            for xx in range(x + w, setw):
                flag = 0
                for yy in range(y + int(h / 3), y + int(h / 1.5)):
                    if img_baw[yy, xx] == 0:
                        right_xy = [xx, yy]
                        flag = 1
                        break
                if flag == 1:
                    break
                if img_baw[y + int(h / 3), xx] == 0:
                    right_xy = [xx, y + int(h / 3)]
                    break
            pointset_word.append([i, left_xy, right_xy])
            cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), 2)
    allarr = []
    # 文本框的标记是0
    for i in pointset_word:
        allarr.append([i, 0, 'NULL'])
    # 指向箭头的标记是1
    for i in pointset_arrow:
        tmp = []
        tmp.append([i[0], 1, i[1]])
        allarr.append(tmp[0])
    # 竖线的标记是2
    for i in pointset_line:
        allarr.append([i, 2, 'NULL'])
    for i in range(len(allarr)):
        for j in range(len(allarr) - 1 - i):
            if md(allarr[j]) > md(allarr[j + 1]):
                tmp = allarr[j + 1]
                allarr[j + 1] = allarr[j]
                allarr[j] = tmp
    for i in range(len(allarr)):
        allarr[i].insert(1, i)
    # print(cvok.weather_line_nr_word(allarr[21], allarr)[0])
    # 表结构：【【坐标,【箭头所在位置坐标（如果是箭头）,左右竖线定位位置（如果是文本）】】，节点编号（包括符号和文本），节点类型，【竖线范围，【竖线起始坐标】，指向数（文本和竖线为null）】，【【前驱指向节点】，【前驱文本节点】】，【【后继指向节点】，【后继文本节点】】】
    arrow_dis_right = []
    arrow_dis_left = []
    specialline=Specialline(allarr)
    for i in allarr:
        if i[2] == 1 and i[1] not in specialline[1]:
            for j in i[0][1]:
                xxx = []
                yyy = []
                dis = 10000
                for k in allarr:
                    if k[2] == 0:
                        line = weather_line_nr_word(k, allarr, specialline[0])
                        ad = minmargin(j, md2(line[0]))
                        if ad[0] < dis:
                            dis = ad[0]
                for k in allarr:
                    if k[2] == 0:
                        line = weather_line_nr_word(k, allarr, specialline[0])
                        ad = minmargin(j, md2(line[0]))
                        if ad[0] == dis:
                            xxx = i
                            yyy = k
                            arrow_dis_right.append([xxx[1], yyy[1]])
            xxx = []
            yyy = []
            dis = 10000
            for k in allarr:
                if k[2] == 0:
                    line = weather_line_nr_word(k, allarr, specialline[0])
                    ad = minmargin(md2(line[1]), i[0][0])
                    if ad[0] < dis:
                        dis = ad[0]
            for k in allarr:
                if k[2] == 0:
                    line = weather_line_nr_word(k, allarr, specialline[0])
                    ad = minmargin(md2(line[1]), i[0][0])
                    if ad[0] == dis:
                        xxx = i
                        yyy = k
                        arrow_dis_left.append([yyy[1], xxx[1]])
    for i in allarr:
        if i[2] == 0:
            i.append([['NULL'],['NULL']])
    for i in arrow_dis_right:
        for j in arrow_dis_left:
            if j[1] == i[0]:
                if allarr[i[1]][4][0] == ['NULL']:
                    allarr[i[1]][4][0] = [j[0]]
                if j[0] not in allarr[i[1]][4][0]:
                    allarr[i[1]][4][0].append(j[0])
                if allarr[j[0]][4][1] == ['NULL']:
                    allarr[j[0]][4][1] = [i[1]]
                if i[1] not in allarr[j[0]][4][1]:
                    allarr[j[0]][4][1].append(i[1])
    allarr=getherword(allarr)
    return allarr

def md(allarr):
    if allarr[1]==1 or allarr[1]==0:
        return allarr[0][0][0]
    return allarr[0][0]
def md2(a):
    if a[2]==2:
        return a[0]
    else:
        return a[0][0]

#写这个方法要把我写死了
def Specialline(allarr):
    '''
    函数功能：找到特殊定位线与文本节点之间的关系
    :param allarr: 所有节点的信息
    :return: 特殊定位线与文本节点之间的关系（如果存在特殊定位线）
    '''
    sl=[]
    needtocancel=[]
    for i in allarr:
        if i[2]==1 and i[3][0]!=0:
            arrowset = []
            x,y,w,h=i[0][0]
            area=x-30,y,30,h
            tmp=[]
            for k in allarr:
                if k[2]==1:
                    for j in k[0][1]:
                        if weather_point_in_area([j[0]+j[2]-1,j[1]+j[3]/2], area):
                            arrowset.append(k[0][0])
                            if k[1] not in needtocancel:
                                needtocancel.append(k[1])
            dis = 10000
            for q in arrowset:
                for p in allarr:
                    if p[2] == 0:
                        margin = minmargin(p[0][0], q)
                        if margin[0] < dis:
                            dis = margin[0]
                            tmp = [i[1], p[1]]
                sl.append(tmp)
        if i[2]==2:
            x, y, w, h = i[0]
            area1=x-30,y,30,h
            area2=x+w,y,30,h
            flag__ = 0
            flag_ = 0
            tmp = []
            arrowset=[]
            nec=[]
            for k in allarr:
                if k[2]==1:
                    if weather_point_in_area([k[0][0][0],k[0][0][1]+k[0][0][3]/2], area2):
                        flag__=1
                    for j in k[0][1]:
                        if weather_point_in_area([j[0]+j[2],j[1]+j[3]/2], area1):
                            arrowset.append(k[0][0])
                            if k[1] not in nec:
                                nec.append(k[1])
                            flag_=1
            if flag__ == 1 and flag_ == 1:
                for qwe in nec:
                    needtocancel.append(qwe)
                dis = 10000
                for q in arrowset:
                    for p in allarr:
                        if p[2] == 0:
                           margin = minmargin(p[0][0], q)
                           if margin[0] < dis:
                               dis = margin[0]
                               tmp = [i[1], p[1]]
                    sl.append(tmp)
    return [sl,needtocancel]

def getherword(allarr):
    '''
    函数功能：根据节点的相关信息判断是否存在同一文本节点被分隔为多个文本节点，如果有则合并
    :param allarr: 所有节点相关信息
    :return: 整合过的节点相关信息
    '''
    cancel=[]
    for i in allarr:
        if i[2]==0:
            for j in allarr:
                if j[2]==0:
                    if i!=j and comparelist(i,j):
                        cancel.append([i[1],j[1]])
                        i[0][0]=getherarea(i[0][0],j[0][0])
                        j[0][0]=[-1,-1,-1,-1]
                        j[4]=[[-1*j[1]],[-1*j[1]]]
    if cancel!=[]:
        print(cancel)
        for i in allarr:
            if i[2]==0:
                k=0
                for j in i[4]:
                    while k<len(j):
                        if j[k] in [x[1] for x in cancel]:
                            del j[k]
                        else:
                            k+=1
    return allarr

def addblankedge(img):
    '''
    函数功能：给目标图片加个白边，为了更好的ocr
    :param img: 目标图片
    :return: 处理后的图片
    '''
    img_cp=img.copy()
    h,w=img_cp.shape
    img=cv2.resize(img,(w+20,h+20))
    for i in img:
        for j in range(len(i)):
            i[j] = 255
    img[10:h+10,10:w+10]=img_cp
    return img

def getherarea(area0,area1):
    '''
    函数功能：合并两个区域（矩形顶点）
    :param area0: 区域信息
    :param area1: 区域信息
    :return: 合并后的信息
    '''
    pointset=[]
    pointset.append([area0[0],area0[1]])
    pointset.append([area0[0]+area0[2], area0[1]+area0[3]])
    pointset.append([area1[0], area1[1]])
    pointset.append([area1[0] + area1[2], area1[1] + area1[3]])
    x,y,w,h=cv2.boundingRect(np.array(pointset))
    return [x,y,w,h]


def findpage(img):
    '''
    函数功能：确定页码
    :param img:待处理图片
    :return:页码字段
    '''
    h,w=img.shape
    img=img[h-int(h/5):h,w-int(w/5):w]
    h, w = img.shape
    imgtmp,pointset=morphological(img, 8, 8, 8, 8)
    tmp=[]
    dis=100000
    for i in pointset:
        if (w-i[0])**2+(h-i[1])**2<dis:
            tmp=i
            dis=i[0]**2+i[1]**2
    img=addblankedge(img[tmp[1]:tmp[1]+tmp[3],tmp[0]:tmp[0]+tmp[2]])
    return pyt.imgtostr(img)

def comparelist(m,n):
    '''
    函数功能：判断两个节点是否可以合并
    :param m: 节点信息
    :param n: 节点信息
    :return: 是否
    '''
    list0=m[4]
    list1=n[4]
    sublist0_0=list0[0]
    sublist0_1 = list0[1]
    sublist1_0 = list1[0]
    sublist1_1=list1[1]
    for i in sublist0_0:
        if i not in sublist1_0:
            return False
    for i in sublist1_0:
        if i not in sublist0_0:
            return False
    for i in sublist0_1:
        if i not in sublist1_1:
            return False
    for i in sublist1_1:
        if i not in sublist0_1:
            return False
    if m[0][0][1]>n[0][0][1] and m[0][0][1]-(n[0][0][1]+n[0][0][3])>50:
        return False
    if n[0][0][1]>m[0][0][1] and n[0][0][1]-(m[0][0][1]+m[0][0][3])>50:
        return False
    return True
from sklearn.externals import joblib
import myw2v
def makenewarr(allarr,img,source):
    '''
    函数功能：按照下方的结构所示构建新的节点信息
    :param allarr:所有节点信息
    :param img: 待处理图片
    :param source: 来源
    :return: 新结构的节点信息
    '''
    page=findpage(img)
    newarr=[]
    # 节点页码 	节点编号	    节点内容    	前节点		后节点		节点类型	   来源     转移条件        备注
    #这两个模型是预先训练好的
    model = myw2v.loadmodel('model_w2v.model')
    clf = joblib.load('svmmodel.model')
    for i in allarr:
        if i[2]==0:
            if i[0][0]!=[-1,-1,-1,-1]:
                bz='null'
                tmp=[]
                tmp.append(page)
                tmp.append(i[1])
                content=pyt.imgtostr(addblankedge(img[i[0][0][1]-1:i[0][0][1] + i[0][0][3]+1, i[0][0][0]-1:i[0][0][0] + i[0][0][2]+1]))
                tmp.append(content)
                tmp.append(i[4][0])
                tmp.append(i[4][1])
                content=myw2v.clearsentence(content)
                content=[i for i in content.lower().split() if
                         i not in nltk.corpus.stopwords.words(
                     'english')]
                content=myw2v.contsentence(content,model)
                y_p = clf.predict([content])
                print(i[1],y_p)
                if y_p[0]==0:
                    tmp.append('action')
                else:
                    tmp.append('condition')
                tmp.append(source)
                tmp.append(bz)
                newarr.append(tmp)
    return newarr

def findend(img_cp,setw):
    '''
    函数功能：找到结构图的上下界
    :param img_cp: 待处理图片
    :param setw: 图片宽度
    :return: 上下界位置
    '''
    img=img_cp.copy()
    thresh,pointset_word_pre=morphological(img, 9, 9, 8, 8)
    margin = setw, setw, setw, setw
    # 找到分界的文本框
    highest = setw
    lowest=0
    for i in pointset_word_pre:
        x, y, w, h = i
        if y > 0.4 * setw and w > 0.7 * setw and y < margin[1]:
            if y <= highest:
                highest = y
        if y+h<0.2*setw and w>0.85*setw:
            if y>=lowest:
                lowest=y+h
    return [lowest,highest]
#encoding=utf-8
def saveastxt(newarr):
    fo=open('tmp.txt','w',encoding='utf8')
    for i in newarr:
        for j in i:
            if type(j)==list:
                for k in j:
                    fo.write(str(k)+',')
                fo.write(';;')
            else:
                fo.write(str(j)+';;')
        fo.write('\n')

import os
def readpath(path):
    outpath=[]
    pathDir = os.listdir(path)
    for i in pathDir:
        tmpdir=os.listdir(path+'/'+i)
        for j in tmpdir:
            outpath.append(path+'/'+i+'/'+j)
    return outpath
import shutil
def copyfile(path,path1):
    pathDir=os.listdir(path)
    for i in pathDir:
        os.mkdir(path1+'/'+i)
        tmpdir = os.listdir(path + '/' + i)
        for j in tmpdir:
            img=cv2.imread(path+'/'+i+'/'+j)
            setw = 1200
            h, w = img.shape[:2]
            img = cv2.resize(img, (setw, int(h / w * setw)))
            h, w = img.shape[:2]
            blured = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(blured, 200, 255, cv2.THRESH_BINARY)
            arrow,line=arrowandline_num(thresh,setw)
            if arrow!=[]:
                shutil.copy(path+'/'+i+'/'+j,path1+'/'+i+'/'+j)

# copyfile('G:/nccn','G:/nccn_aft')