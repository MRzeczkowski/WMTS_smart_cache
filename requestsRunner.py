import json
import requests
import time
import pandas as pd
import re
import random

if __name__ == '__main__':

    files = [
        '/Users/mateusz/Documents/projects/WMTS_smart_cache/proxy_data/statistics/getTileStatistics_2022-12-03 17:47:47.978605.csv'
    ]

    for file in files:

        df = pd.read_csv(file)

        lastRequestedTime = 0

        for index, row in df.iterrows():

            currentRequestedTime = row['requested_time']

            if not lastRequestedTime == 0:
                waitTime = min(abs(currentRequestedTime -
                               lastRequestedTime), random.randint(2, 3) * 60)
                print("waiting for {0}s".format(waitTime))
                time.sleep(waitTime)

            lastRequestedTime = currentRequestedTime

            tilePosition = row['tile_position']

            indexes = list(map(int, re.findall(r'\d+', tilePosition)))

            matrixSideLength = (2 ** indexes[0]) - 1

            if indexes[1] <= matrixSideLength and indexes[1] >= 0:
                indexes[1] += random.randint(-1, 1)
                if indexes[1] > matrixSideLength:
                    indexes[1] = matrixSideLength
                elif indexes[1] < 0:
                    indexes[1] = 0

            if indexes[2] <= matrixSideLength and indexes[2] >= 0:
                indexes[2] += random.randint(-1, 1)
                if indexes[2] > matrixSideLength:
                    indexes[2] = matrixSideLength
                elif indexes[2] < 0:
                    indexes[2] = 0

            url = 'http://localhost:8184/USGS/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=San_Francisco_2015&STYLE=default&FORMAT=image/jpeg&TILEMATRIXSET=San_Francisco&TILEMATRIX={0}&TILEROW={1}&TILECOL={2}'.format(
                *indexes)
            headers = json.loads(
                '{"Accept": "*/*", "User-Agent": "Mozilla/5.0 QGIS/32212", "Connection": "Keep-Alive", "Accept-Encoding": "gzip, deflate", "Accept-Language": "en-US,*"}')

            requests.request(
                method="GET",
                url=url,
                headers=headers,
                data=None,
                cookies=None,
                allow_redirects=False)

print('done!')
