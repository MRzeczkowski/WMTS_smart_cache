from flask import Flask, request, Response
import requests, os, json, sqlite3, re, matplotlib, io, numpy as np, matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator

matplotlib.use('SVG')

app = Flask(__name__)

SITE_NAME = 'http://localhost:8183/'
DB_NAME = 'tiles.db'

def extractTileInfo(tileMatrixId):
    select = lambda cur: cur.execute(
        "SELECT tileRow, tileCol, countedRequests FROM tiles WHERE tileMatrix = ?", (tileMatrixId,)).fetchall()

    return executeOnDatabase(select)

@app.route('/tileMatrixHeatmap/<int:tileMatrixId>')
def getTileMatrixHeatmap(tileMatrixId):
    tileInfo = extractTileInfo(tileMatrixId)

    width = max(map(lambda row: row[0], tileInfo)) + 1
    length = max(map(lambda row: row[1], tileInfo)) + 1

    data = np.zeros(shape=(width, length))

    for row in tileInfo:
        data[row[0], row[1]] = row[2]

    fig, ax = plt.subplots()

    if length == 1 and width == 1:
        plt.axis('off')

    else:
        maxBins = 20

        maxXBins = maxBins
        maxYBins = maxBins

        if width < maxXBins:
            maxXBins = width

        if length < maxYBins:
            maxYBins = length

        plt.locator_params(axis='x', nbins=maxXBins)
        plt.locator_params(axis='y', nbins=maxYBins)

    im = ax.imshow(data, aspect='equal')
    
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def incrementCountedRequestsForTile(tilePosition):
    update = lambda cur: cur.execute(
        "UPDATE tiles SET countedRequests = countedRequests + 1 WHERE tileMatrix = ? AND tileRow = ? AND tileCol= ?", tilePosition)

    executeOnDatabase(update)

@app.route('/<path:path>')
def proxy(path):
    global SITE_NAME
    global DB_NAME

    newUrl = request.url.replace('%2F', '/')

    newUrlLower = newUrl.lower()

    isGetCapabilitiesRequest = 'getcapabilities' in newUrlLower

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

    if 'gettile' in newUrlLower:

        tileMatrix = re.search(r'tilematrix=(\d*)', newUrlLower).group(1)
        tileRow = re.search(r'tilerow=(\d*)', newUrlLower).group(1)
        tileCol = re.search(r'tilecol=(\d*)', newUrlLower).group(1)

        incrementCountedRequestsForTile((tileMatrix, tileRow, tileCol))

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

def executeOnDatabase(func):
    con = sqlite3.connect(DB_NAME)
    cur = con.cursor()

    result = func(cur)

    con.commit()
    cur.close()

    return result   

def setupTilesDatabase(tileMatrixCount):
 
    def setupAndSeedTilesTable(cur):
        cur.execute("CREATE TABLE tiles(tileMatrix INTEGER, tileRow INTEGER, tileCol INTEGER, countedRequests INTEGER)")
        cur.execute("CREATE UNIQUE INDEX tilePosition ON tiles(tileMatrix, tileRow, tileCol)")

        for tileMatrix in range(tileMatrixCount):
            matrixSize = pow(2, tileMatrix)
            for tileRow in range(matrixSize):
                for tileCol in range(matrixSize):
                    tilePosition = (tileMatrix, tileRow, tileCol)
                    cur.execute("INSERT INTO tiles VALUES (?, ?, ?, 0)", tilePosition)

    executeOnDatabase(setupAndSeedTilesTable)

if __name__ == '__main__':
    if not os.path.isfile(DB_NAME):
        setupTilesDatabase(11)

    app.run(debug = False, port=9000)
