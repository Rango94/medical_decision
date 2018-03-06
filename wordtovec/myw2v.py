import gensim
import nltk.corpus
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.preprocessing import Imputer
import sys
sys.path.append('E:\libsvm-3.22\python')
from svmutil import *
def contsentence(sentence,model):
    out=0
    cont=0
    for i in sentence:
        cont+=1
        try:
            out=out+model.wv[i.lower()]
        except:
            # print('跳过')
            continue
    return out/max(cont,1)
def trainmodel(filename,Stopwords=True):
    print('整理格式')
    with open(filename,encoding='utf8') as fo:
        if Stopwords:
            sentences=[[i for i in clearsentence(line.replace('\n', '').lower()).split() if i not in nltk.corpus.stopwords.words('english')] for line in fo if line != '\n']
            # sentences=[[i for i in line.replace('\n', '').lower().split() if
            #             (i.isalpha() or i.isdigit() or i=='>' or i=='<'or i=='±') and i not in nltk.corpus.stopwords.words('english')] for line in fo if line != '\n']
        else:
            sentences = [[i for i in line.replace('\n', '').lower().split() if
                          i.isalpha()] for line in fo if line != '\n']
    print("开始训练")
    model = gensim.models.Word2Vec(sentences, size=100, min_count=5)
    print('模型已经产生')
    return model
def savemodel(filename,modelname):
    modelname.save(filename)
    print('baocun')
def loadmodel(filename):
    model = gensim.models.Word2Vec.load(filename)
    return model
def Readfile_(filename):
    out=[]
    with open(filename,encoding='utf8') as fo:
        file=fo.readlines()
        for line in file:
            line=clearsentence(line)
            line=line.replace('\n','').replace('\t','').split(' ')
            tmp=[i for i in line if i not in nltk.corpus.stopwords.words('english') and i!='']
            if tmp!=[]:
                out.append(tmp)
    return out
def builddata(filename1,filename2,flag1,flag2,model):
    sentences1=Readfile_(filename1)
    sentences2=Readfile_(filename2)
    L2 = np.empty((100, ))
    L1 = np.empty((100, ))
    for line in sentences1:
        try:
            tmp=contsentence(line,model)
            L1=np.vstack((L1,tmp))
        except:
            continue
    for line in sentences2:
        try:
            tmp=contsentence(line,model)
            L2=np.vstack((L2,tmp))
        except:
            continue
    X=np.vstack((L1,L2))
    if flag1==0 and flag2==1:
        Y=np.hstack((np.zeros((len(L1))),np.ones((len(L2)))))
    else:
        Y = np.hstack((np.ones((len(L1))), np.zeros((len(L2)))))
    for i in range(int(len(Y)/2)):
        swap1=int(len(Y)*random.random())
        swap2=int(len(Y)*random.random())
        tmp=X[swap1,:].copy()
        X[swap1,:]=X[swap2,:].copy()
        X[swap2,:]=tmp.copy()
        tmp=Y[swap1]
        Y[swap1]=Y[swap2]
        Y[swap2]=tmp
    return X,Y

def PSO(n,model):
    sc = StandardScaler()
    canshu=np.zeros((2,n))
    canshu[0,0]=6.18514006143
    canshu[1,0]=0.0226782206741
    bestc=canshu[0,0]
    bestg=canshu[1,0]
    bestsinglecv=np.zeros((1,n))
    bestsinglec=np.zeros((1,n))
    bestsingleg=np.zeros((1,n))
    bestcv=0
    v=np.zeros((2,n))
    for i in range(1,n):
        canshu[0,i]=0.2+10*random.random()
        canshu[1,i]=0.5*random.random()
    while 1==1:
        for i in range(0,n):
            c=canshu[0,i]
            g=canshu[1,i]
            print("bestc:",bestc)
            print("bestg:",bestg)
            print("bestcv:",bestcv)
            print('-----------------')
            p_acc=0
            for q in range(5):
                x, y = builddata('action.txt', 'condition.txt', 0, 1, model)
                margin = int(len(y) / 2)
                start=0
                clf = svm.SVC(C=c, gamma=g)
                x = Imputer().fit_transform(x)
                # x=sc.fit_transform(x)
                clf.fit(x[start:margin,:], y[start:margin])
                Y_=clf.predict(x[margin:,:])
                for k in range(len(Y_)):
                    if y[k+margin] == Y_[k]:
                        Y_[k] = 1
                    else:
                        Y_[k] = 0
                p_acc+=np.mean(Y_)
            p_acc/=5
            cv=p_acc
            if cv>=bestsinglecv[0,i]:
                bestsinglecv[0,i] = cv
                bestsinglec[0,i] = c
                bestsingleg[0,i] = g
                clf = svm.SVC(C=c, gamma=g)
                clf.fit(x, y)
                joblib.dump(clf, 'svmmodel.model')

        print('bestsinglecv:',bestsinglecv)
        print('bestsinglec:',bestsinglec)
        print('bestsingleg:',bestsingleg)
        bestcv = max(bestsinglecv[0,:])
        index=np.argmax(bestsinglecv)
        bestc=bestsinglec[0,index]
        bestg=bestsingleg[0,index]
        for i in range(n):
            v[0,i]=0.5*v[0,i]+random.random()*(bestsinglec[0,i]-canshu[0,i])+random.random()*(bestc-canshu[0,i])
            if v[0,i]>=0.5:
                v[0,i]=0.5
            elif v[0,i]<=-0.5:
                v[0,i]=-0.5
            canshu[0,i]=canshu[0,i]+v[0,i]
            v[1,i]=0.5*v[1,i]+random.random()*(bestsingleg[0,i]-canshu[1,i])+random.random()*(bestg-canshu[1,i])
            if v[1,i]>=0.5:
                v[1,i]=0.5
            elif v[1,i]<=-0.5:
                v[1,i]=-0.5
            canshu[1,i]=max(canshu[1,i]+v[1,i],0.00001)
        print(canshu)
from sklearn.externals import joblib

def testsvmmodel(modelname,c,g,XX,YY):
    print(len(XX))
    XX=Imputer().fit_transform(XX)
    print(len(XX))
    X=XX[:int(2*len(YY)/3),:]
    Y=YY[:int(2*len(YY)/3)]
    X_=XX[int(2*len(YY)/3):,:]
    Y_=YY[int(2*len(YY)/3):]
    # clf = svm.SVC(C=c, gamma=g)
    # clf.fit(X,Y)
    clf = joblib.load(modelname)
    y_p=clf.predict(XX)
    print(y_p)
    for i in range(len(y_p)):
        if YY[i]==y_p[i]:
            y_p[i]=1
        else:
            y_p[i]=0
    print(np.mean(y_p))

def clearsentence(sentence):
    out=''
    for i in sentence:
        if i.isalpha() or i.isdigit()  or i==' ':
            out+=i
        if  i=='>' or i=='<' or i=='±':
            out+=' '+i+' '
    out=' '.join(out.split())
    # print(out)
    return out
