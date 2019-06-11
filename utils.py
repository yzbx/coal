import time
import json
def detection(data_json):
    data=json.loads(data_json)
    for i in range(100):
        print(i,data)
        time.sleep(1)

    return 0