import os
from tqdm import tqdm

INPUT_DIR = "./Data/"
OUTPUT_DIR = "./Preprocessed/"
START_YEAR = 1996
END_YEAR = 2006


def preprocess(filename: str) -> None:
    data = {}
    with open(os.path.join(os.path.join(INPUT_DIR, filename)), "r", encoding="UTF-8") as fp:
        text = fp.read().strip().split(' ')
        for item in text:
            if ((item == "") or (item == "\n") or (item == " ")):
                raise
            if (item not in data):
                data[item] = 0
            data[item] += 1
    with open(os.path.join(os.path.join(OUTPUT_DIR, filename)), "w", encoding="UTF-8") as fp:
        for key in data:
            print(key, data[key], file=fp)
    return None


if __name__ == "__main__":
    for year in range(START_YEAR, END_YEAR + 1):
        os.makedirs(os.path.join(OUTPUT_DIR, "%d.tok/" % year), exist_ok=True)
        filelist = os.listdir(os.path.join(INPUT_DIR, "%d.tok/" % year))
        for filename in tqdm(filelist):
            preprocess("./%d.tok/%s" % (year, filename))

    pass
