from flask import Flask, request, Response
import requests, os, json, sqlite3, re, matplotlib, io, numpy as np, matplotlib.pyplot as plt, time, csv
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

matplotlib.use('SVG')

app = Flask(__name__)

SlidingWindowLengthInSeconds = 2 * 60 * 60

SITE_NAME = 'http://localhost:8183/'
DB_NAME = 'tiles.db'

def extractTileInfo(statiticType, tileMatrixId):
    select = lambda cur: cur.execute(
        "SELECT tileRow, tileCol, {0} FROM tiles WHERE tileMatrix = ?".format(statiticType), (tileMatrixId,)).fetchall()

    return executeOnDatabase(select)

@app.route('/tileMatrixHeatmap/<string:statiticType>/<int:tileMatrixId>')
def getTileMatrixHeatmap(statiticType, tileMatrixId):

    availableStatisticsTypes = ['countedRequests', 'size', 'recency', 'frequency', 'target']

    if statiticType not in availableStatisticsTypes:
        return 'Incorrect statistic type. Chose one of the following: ' + ', '.join(availableStatisticsTypes)

    tileInfo = extractTileInfo(statiticType, tileMatrixId)

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

def updateStatisticsForTile(tilePosition, tileSizeInBytes):

    requestUnixTimestamp = int(time.time())

    select = lambda cur: cur.execute("""SELECT countedRequests, frequency, lastRequestedUnixTimestamp 
        FROM tiles 
        WHERE tileMatrix = ? AND tileRow = ? AND tileCol= ?""", tilePosition).fetchone()

    lastRequestStatistics = executeOnDatabase(select)

    countedRequests = lastRequestStatistics[0]
    frequency = lastRequestStatistics[1]
    lastRequestedUnixTimestamp = lastRequestStatistics[2]

    timeSinceLastRequest = requestUnixTimestamp - lastRequestedUnixTimestamp

    if timeSinceLastRequest <= SlidingWindowLengthInSeconds:
        frequency += 1
    else:
        frequency = round(max(frequency / (timeSinceLastRequest/SlidingWindowLengthInSeconds), 1))

    recency = 0.0

    if countedRequests == 0:
        recency = SlidingWindowLengthInSeconds
    else:
        recency = max(SlidingWindowLengthInSeconds, timeSinceLastRequest)

    countedRequests += 1

    mostRecentStatistics = (tileSizeInBytes, frequency, recency, countedRequests, requestUnixTimestamp)

    update = lambda cur: cur.execute(
        """UPDATE tiles 
            SET
                size = ?,
                frequency = ?,
                recency = ?,
                countedRequests = ?,
                lastRequestedUnixTimestamp = ?
            WHERE tileMatrix = ? AND tileRow = ? AND tileCol= ?""",
            (mostRecentStatistics + tilePosition))

    executeOnDatabase(update)

    statisticsFileName = 'getTileStatistics.csv'

    fileExsists = os.path.isfile(statisticsFileName)

    with open(statisticsFileName, 'a', newline=os.linesep) as csvfile:
        fieldnames = ['tile_position', 'requested_time', 'size', 'frequency', 'recency', 'target']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not fileExsists:
            writer.writeheader()

        writer.writerow(
            {'tile_position': tilePosition,
            'requested_time': requestUnixTimestamp, 
            'size': tileSizeInBytes, 
            'frequency': frequency, 
            'recency': recency, 
            'target': 0})

@app.route('/<path:path>')
def proxy(path):
    global SITE_NAME

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

    newUrl = newUrl.replace(request.host_url, SITE_NAME)

    resp = requests.request(
        method=method,
        url=newUrl,
        headers=headers,
        data=data,
        cookies=cookies,
        allow_redirects=False)

    newContent = resp.content

    if 'gettile' in newUrlLower:

        tileMatrix = re.search(r'tilematrix=(\d*)', newUrlLower).group(1)
        tileRow = re.search(r'tilerow=(\d*)', newUrlLower).group(1)
        tileCol = re.search(r'tilecol=(\d*)', newUrlLower).group(1)

        updateStatisticsForTile((tileMatrix, tileRow, tileCol), len(newContent))

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
        cur.execute("""CREATE TABLE tiles(
            tileMatrix INTEGER, 
            tileRow INTEGER, 
            tileCol INTEGER, 
            countedRequests INTEGER DEFAULT 0,
            size INTEGER DEFAULT 0,
            recency INTEGER DEFAULT 0,
            frequency REAL DEFAULT 0,
            target REAL DEFAULT 0,
            lastRequestedUnixTimestamp INTEGER DEFAULT 0)""")
        cur.execute("CREATE UNIQUE INDEX tilePosition ON tiles(tileMatrix, tileRow, tileCol)")

        for tileMatrix in range(tileMatrixCount):
            matrixSize = pow(2, tileMatrix)
            for tileRow in range(matrixSize):
                for tileCol in range(matrixSize):
                    tilePosition = (tileMatrix, tileRow, tileCol)
                    cur.execute("INSERT INTO tiles VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0)", tilePosition)

    executeOnDatabase(setupAndSeedTilesTable)

if __name__ == '__main__':
    if not os.path.isfile(DB_NAME):
        setupTilesDatabase(11)

    app.run(debug = False, port=9000)
