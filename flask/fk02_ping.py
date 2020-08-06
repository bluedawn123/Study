from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello333():
    return "<h1>dfdfdfdf</h1>"

@app.route('/ping', methods = ['GET'])
def ping():
    return "<h1>dfdfsfsdfsdfdfdfdf</h1>"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
