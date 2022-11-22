import json, requests, time

if __name__ == '__main__':
    with open("./requestLog.txt", "r") as requestLog:
        for line in requestLog.readlines():
            time.sleep(10)
            jsonObject = json.loads(line)
            requests.request(
                method= jsonObject['method'],
                url=jsonObject['url'],
                headers=jsonObject['headers'],
                data=jsonObject['data'].encode(),
                cookies=jsonObject['cookies'],
                allow_redirects=False)