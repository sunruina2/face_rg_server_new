from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect', namespace='/test_conn')
def test_message():
    print('client connect to test_conn')


@socketio.on('client response', namespace='/test_conn')
def test_message(message):
    print(message)
    while 1:
        time.sleep(2)
        emit('server response', {'data': 'server response msg'})


if __name__ == '__main__':
    socketio.run(app)
