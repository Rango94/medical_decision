con=open('condition.txt',encoding='utf8')
act=open('action.txt',encoding='utf8')
con=con.readlines()
act=act.readlines()
conarr={}
actarr={}
allarr={}
for i in con:
    i=i.lstrip(' ').replace('\n','')
    arr=i.split(' ')
    for j in arr:
        if j.isalpha():
            if j not in conarr:
                conarr[j]=1
            if j in conarr:
                conarr[j]+=1
            if j not in allarr:
                allarr[j]=1
            else:
                allarr[j]+=1
for i in act:
    i=i.lstrip(' ').replace('\n','')
    arr=i.split(' ')
    for j in arr:
        if j.isalpha():
            if j not in actarr:
                actarr[j]=1
            if j in actarr:
                actarr[j]+=1
            if j not in allarr:
                allarr[j]=1
            else:
                allarr[j]+=1
print(allarr)
print(conarr)
print(actarr)
