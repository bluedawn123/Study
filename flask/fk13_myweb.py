# 접속 : odbc 방식

import pyodbc as pyo

# print("우갸갹")

server = '127.0.0.1'
database = 'bitdb'
username = 'bit2'
password = '12345'

conn = pyo.connect('DRIVER={ODBC Driver 17 for SQL Server};' + f'SERVER={server};' +
                   'PORT=1433;' +  f'DATABASE={database};' + f'UID={username};' + f'PWD={password};')

curser = conn.cursor()

tsql = 'SELECT * FROM sonar;'
'''
with curser.execute(tsql):
    row = curser.fetchone()

    while row :
        print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " +
              str(row[3]) + " " + str(row[4]))

        row = curser.fetchone()
'''
#플라스크 코딩
from flask import Flask, render_template
app = Flask(__name__)

@app.route('/sqltable')
def showsql():
    curser.execute(tsql)
    return render_template ("myweb.html", rows = curser.fetchall())

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

conn.close()
