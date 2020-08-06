import sqlite3

conn = sqlite3.connect('test.db')
cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT,
               FoodName TEXT, Company TEXT, Price INTEGER)""")

sql = "DELETE FROM supermarket"
cursor.execute(sql)



# 데이터 넣기
sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?, ?, ?, ?, ?)"
cursor.execute(sql, (1, '똥', '오줌', '마트', 1500))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?, ?, ?, ?, ?)"
cursor.execute(sql, (2, '물', '가레', '편의점', 1200))

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?, ?, ?, ?, ?)"

cursor.execute(sql, (1, '피', '눈알', 'ㅈ트캠프', 11500))

sql = "SELECT * FROM supermarket"
#sql = "SELECT Itemno, Category, FoodName, Company, Price FROM supermarket"

cursor.execute(sql)

rows = cursor.fetchall()

for row in rows:
   print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " +
         str(row[3]) + " " + str(row[4]))



conn.commit()


conn.close()