######## Libraries:

from flask import Flask

### Initialize the app:
app = Flask(__name__)

#Define a route of hello world:

@app.route('/')
def hello_world():
    return 'Hello world!'

app.run(debug=True, port=8085)
