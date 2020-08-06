import pymssql as ms

conn = ms.connect(server = '127.0.0.1', user = 'bit2', password='12345', database='bitdb')

cursor = conn.cursor()  #cursor 은 지정한다

cursor.execute("SELECT * FROM sonar;")

row = cursor.fetchone()   #한 줄을 갖고 올거다.

while row :
    print("첫칼럼 :  %s, 둘 컬럼 : %s" %(row[0], row[1]))
    row = cursor.fetchone()

conn.close()
