from flask import Flask, request, Response
import requests, os, json

app = Flask(__name__)

SITE_NAME = 'http://localhost:8183/'

@app.route('/')
def index():
    return 'Flask is running!'

@app.route('/<path:path>')

def proxy(path):
    global SITE_NAME

    newUrl = request.url.replace('%2F', '/')

    isGetCapabilitiesRequest = 'getcapabilities' in newUrl.lower()

    method = request.method
    headers = {key: value for (key, value) in request.headers if key != 'Host'}
    data = request.get_data()
    cookies = request.cookies

    if not isGetCapabilitiesRequest:
        dictionary = {
            "url": newUrl,
            "method": method,
            "headers": headers,
            "data": data.decode("utf-8"),
            "cookies": cookies
        }

        with open("./requestLog.txt", "a") as requestLog:
            json.dump(dictionary, requestLog)
            requestLog.write(os.linesep)

    newUrl = newUrl.replace(request.host_url, SITE_NAME)

    resp = requests.request(
        method=method,
        url=newUrl,
        headers=headers,
        data=data,
        cookies=cookies,
        allow_redirects=False)

    newContent = resp.content

    if isGetCapabilitiesRequest:
        newContent = resp.content.decode().replace(SITE_NAME, request.host_url)

    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]

    response = Response(newContent, resp.status_code, headers)
    return response
    
if __name__ == '__main__':
    app.run(debug = False, port=9000)