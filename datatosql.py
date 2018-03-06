import mysql.connector as mc
conn=mc.connect(
        host='172.16.18.43',
        port=3306,
        user='dev09',
        passwd='00000000',
        db='test'
    )
cur = conn.cursor()
def createtable(tablename):
    cur.execute('CREATE TABLE '+tablename +'(page TEXT, num INT, content TEXT, father TEXT, son TEXT, type TEXT, remarks TEXT )')
def adddatatosql(address):
    fo=open(address,encoding='utf8')
    # cur.execute(
    #     """insert into predata (page,num,content,father,son,type,remarks) values('line[0]','3','line[2]','line[3]','line[4]','line[5]','line[6]')""")
    while 1:
        line=fo.readline()
        if not line:
            break
        line=line.split(';;')
        cur.execute(
            ' INSERT INTO predata (page,num,content,father,son,type,remarks) VALUES(%s,%s,%s,%s,%s,%s,%s);',[line[0],int(line[1]),line[2],line[3],line[4],line[5],line[6]])
        conn.commit()
def deletetable(tablename):
    cur.execute(
        'DROP TABLE ' + tablename+';')


def refreshtable(tablename):
    deletetable(tablename)
    createtable(tablename)
# createtable('predata')
