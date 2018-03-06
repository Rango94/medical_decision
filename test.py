with open('C:/Users\wangn\Desktop/1111.xml',encoding='utf8') as fo:
    fo=fo.readlines()
    add=0
    for i in range(36):
        k=fo[i].split(',')
        add+=len(k)
    print(add)

