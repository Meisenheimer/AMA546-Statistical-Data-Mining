import os
import numpy as np


def words(base_dir: str):
    data = {
        "Lasso": {},
        "Ridge": {},
        "DecisionTree": {},
    }

    filelist = os.listdir(base_dir)

    n = 1000

    for key in data:
        for folder in filelist:
            if (key not in folder):
                continue
            filename = os.path.join(base_dir, folder, "Vocabulary.txt")
            with open(filename, encoding="UTF-8") as fp:
                text = fp.readlines()

            ssid = folder.split('_')
            ssid = int(ssid[1])

            data[key][ssid] = [[], [], [], []]
            for item in text:
                item = item.strip().split(' ')
                data[key][ssid][0].append(item[0])
                data[key][ssid][1].append(float(item[1]))

            index = list(np.argsort(data[key][ssid][1]))[::-1]
            if (np.min(data[key][ssid][1]) == np.max(data[key][ssid][1])):
                print(key, ssid)
                print(len(data[key][ssid][1]))
                print([data[key][ssid][1][i] for i in index[:n]])
                print([data[key][ssid][1][i] for i in index[-n:]])
                raise
            data[key][ssid][2] = set([data[key][ssid][0][i] for i in index[:n]])
            data[key][ssid][3] = set([data[key][ssid][0][i] for i in index[-n:]])

    year = list(range(1996, 2007 - int(base_dir[-1])))
    for k in data:
        print(k)
        for y1 in year[:-1]:
            y2 = y1 + 1
            print(f"{y1}-{y2}",
                  100 * len(data[k][y1][2] & data[k][y2][2]) / n,
                  100 * len(data[k][y1][3] & data[k][y2][3]) / n,
                  100 * len((data[k][y1][2] & data[k][y2][3]) or (data[k][y1][3] & data[k][y2][2])) / (2 * n))
        print("")

    for k1 in data:
        for k2 in data:
            if (k1 >= k2):
                continue
            print(k1, k2)
            for y1 in year:
                print(f"{y1}",
                      100 * len(data[k1][y1][2] & data[k2][y1][2]) / n,
                      100 * len(data[k1][y1][3] & data[k2][y1][3]) / n,
                      100 * len((data[k1][y1][2] & data[k2][y1][3]) or (data[k1][y1][3] & data[k2][y1][2])) / (2 * n))
            print("")

    return None


if __name__ == "__main__":
    words("Res-1")
    # words("Res-3")
