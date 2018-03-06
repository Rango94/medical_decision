judge=open('judge.txt',encoding='utf8')
con=open('condition.txt','w',encoding='utf8')
act=open('action.txt','w',encoding='utf8')
fo1=open('1111.txt',encoding='utf8')
flag=0
condition=[]
action=[]
xxxx=judge.readlines()
for line in xxxx:
    start = max(line.find('"original_en": "'),line.find('"original_en":"'))
    if start!=-1:
        end=len(line)-1
        if start!=-1:
            end=line[start+len('"original_en": "'):].find('"')
            str=line[start+len('"original_en": "')-1:end+len('"original_en": "')+8].replace("\n",'')
            str=str.split('→')
            for i in str:
                condition.append(i.replace('"',''))
yyyy=fo1.readlines()
flag = 0
tmp=''
for line in yyyy:
    add=line.find('"')

    if line[add+1:].find('"')!=-1 and flag==0:
        action.append(line.replace('\n','').replace('"',''))
        continue
    if add==-1 and flag==0:
        action.append(line.replace('\n','').replace('"',''))
        continue
    if add!=-1 and flag==0:
        tmp = ''
        tmp=line
        flag=1
        continue
    if add==-1 and flag==1:
        tmp=tmp+' '+line
        continue
    if add!=-1 and flag==1:
        flag=0
        tmp=tmp+' '+line
        print(tmp)
        action.append(tmp.replace('\n','').replace('"',''))
        continue
for i in action:
    if i!='':
        act.write(i+'\n')
for i in condition:
    if i!='':
        con.write(i+'\n')
print(action)
print(condition)

