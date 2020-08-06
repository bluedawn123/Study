from flask import Flask, request
from flask import render_template
import sqlite3

app = Flask(__name__)

#db생성
conn = sqlite3.connect("./data/wanggun.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM general")

print(cursor.fetchall())

@app.route('/')
def run():
    conn = sqlite3.connect('./data/wanggun.db')
    c.execute("SELECT * FROM general ")

    

