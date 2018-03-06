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
    cur.execute('CREATE TABLE '+tablename +'(page TEXT, num INT, content TEXT, father TEXT, sun TEXT, type TEXT, remarks TEXT )')
def adddatatosql(address):
    cur.execute(
        ' LOAD DATA LOCAL INFILE "'+address+'" INTO TABLE predata; ')
