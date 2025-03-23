import os
from matplotlib import pyplot as plt


def draw(base_dir: str, key: str):
    tmp = {}

    filelist = os.listdir(base_dir)
    n = len(filelist) // 3
    for folder in filelist:
        filename = os.path.join(base_dir, folder, "log.txt")
        with open(filename, encoding="UTF-8") as fp:
            text = fp.readlines()

        ssid = folder.split('_')
        ssid = '-'.join([ssid[i] for i in [0, 1, 2]])

        tmp[ssid] = {}
        for i in range(len(text)):
            if (key in text[i]):
                text[i] = text[i].strip().replace('/', ' ').split(' ')
                if ("Mean" in text[i]):
                    tmp[ssid]["test"] = (text[i][3].strip(",."), text[i][4].strip(",."))
                if ("Predict:" in text[i]):
                    tmp[ssid]["pred"] = (text[i][3].strip(",."), text[i][4].strip(",."))

    for item in tmp:
        print(item, tmp[item])

    data = {
        "Naive_test": [0.0] * n,
        "Naive_pred": [0.0] * n,
        "Lasso_test": [0.0] * n,
        "Lasso_pred": [0.0] * n,
        "Ridge_test": [0.0] * n,
        "Ridge_pred": [0.0] * n,
        "DecisionTree_test": [0.0] * n,
        "DecisionTree_pred": [0.0] * n,
    }

    for item in ["Lasso", "Ridge", "DecisionTree"]:
        for year in range(1996, 2007 - int(base_dir[-1])):
            ssid = "%s-%d-%d" % (item, year, year + int(base_dir[-1]) - 1)
            data["Naive_test"][year - 1996] = float(tmp[ssid]["test"][1])
            data["Naive_pred"][year - 1996] = float(tmp[ssid]["pred"][1])
            data[item + "_test"][year - 1996] = float(tmp[ssid]["test"][0])
            data[item + "_pred"][year - 1996] = float(tmp[ssid]["pred"][0])
    print("")
    for item in data:
        print(item, data[item])

    label = []
    for year in range(1996, 2007 - int(base_dir[-1])):
        if (base_dir[-1] == '1'):
            label.append("%s" % year)
        else:
            label.append("%s-%s" % (year, year + int(base_dir[-1]) - 1))

    plt.clf()
    plt.figure(figsize=(10, 3))
    plt.grid()
    plt.plot(range(n), data["Naive_test"], label="Naive")
    plt.plot(range(n), data["Lasso_test"], label="Lasso")
    plt.plot(range(n), data["Ridge_test"], label="Ridge")
    plt.plot(range(n), data["DecisionTree_test"], label="DT")
    plt.xlabel("Training Data (year)")
    plt.ylabel(key + " Test")
    plt.xticks(ticks=range(n), labels=label)
    plt.legend()
    plt.savefig("%s_%s_test.jpg" % (base_dir, key), dpi=720, bbox_inches="tight")
    plt.close()

    plt.clf()
    plt.figure(figsize=(10, 3))
    plt.grid()
    plt.plot(range(n), data["Naive_pred"], label="Naive")
    plt.plot(range(n), data["Lasso_pred"], label="Lasso")
    plt.plot(range(n), data["Ridge_pred"], label="Ridge")
    plt.plot(range(n), data["DecisionTree_pred"], label="DT")
    plt.xlabel("Training Data (year)")
    plt.ylabel(key + " Predict")
    plt.xticks(ticks=range(n), labels=label)
    plt.legend()
    plt.savefig("%s_%s_pred.jpg" % (base_dir, key), dpi=720, bbox_inches="tight")
    plt.close()

    return None


if __name__ == "__main__":
    draw("Res-1", "MAE")
    draw("Res-1", "MSE")
    draw("Res-3", "MAE")
    draw("Res-3", "MSE")
