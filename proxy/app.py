from flask import Flask,request,redirect,Response
import requests

app = Flask(__name__)

SITE_NAME = 'http://localhost:8183/'

@app.route('/')
def index():
    return 'Flask is running!'

@app.route('/<path:path>')

def proxy(path):
    global SITE_NAME

    newUrl = request.url.replace(request.host_url, SITE_NAME).replace('%2F', '/')

    resp = requests.request(
        method=request.method,
        url=newUrl,
        headers={key: value for (key, value) in request.headers if key != 'Host'},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False)

    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]

    response = Response(resp.content, resp.status_code, headers)
    return response
    
if __name__ == '__main__':
    app.run(debug = False, port=9000)