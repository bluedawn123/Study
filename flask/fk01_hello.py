from flask import Flask

app  = Flask(__name__)

@app.route('/')
def hello333():
    return "<h1> hello my name is </h1>"

@app.route('/bit')
def hello334():
    return "<h1>hello bit <h1/>"

@app.route('/bit/bitcamp')
def hello334():
    return "<h1>helsssssssssssssssssslo bit <h1/>"

@app.route('/bit')
def hello334():
    return "<h1>hello bit <h1/>"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8888, debug=True)