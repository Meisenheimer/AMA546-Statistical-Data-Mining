import os
from tqdm import tqdm

DATA_DIR = "../Data/Data/"
PRE_DATA_DIR = "../Data/Preprocessed/"


def load_logvol(year: int) -> dict:
    """
    Load {year}.logvol.-12.txt and {year}.logvol.+12.txt, then output the data in a dict {key: (preceding year, following year)}.
    """
    with open(os.path.join(DATA_DIR, "%d.logvol.-12.txt" % year), "r", encoding="UTF-8") as fp:
        text_1 = fp.readlines()
    with open(os.path.join(DATA_DIR, "%d.logvol.+12.txt" % year), "r", encoding="UTF-8") as fp:
        text_2 = fp.readlines()
    data = {}
    if (len(text_1) != len(text_2)):
        raise
    for i in range(len(text_1)):
        tmp_1 = text_1[i].replace("\n", "").split(' ')
        tmp_2 = text_2[i].replace("\n", "").split(' ')
        if ((len(tmp_1) != 2) or (len(tmp_2) != 2) or (tmp_1[1] != tmp_2[1])):
            raise
        data[tmp_1[1]] = (float(tmp_1[0]), float(tmp_2[0]))
    return data


def load_all_logvol(start_year: int, end_year: int) -> dict:
    """
    Load and merge all logvol files with year in [args.start_year, args.end_year].
    """
    logvol = {}
    for year in tqdm(range(start_year, end_year + 1)):
        logvol.update(load_logvol(year))
    return logvol


def text2dict(text: str):
    """
    Convert the text into dict {words: count}.
    """
    data = {}
    for item in text.split(' '):
        if (("" == item) or ("\n" == item) or (" " == item)):
            raise
        if (item not in data):
            data[item] = 0
        data[item] += 1
    return data


def load_tok(year: int) -> dict:
    """
    Load all files under {year}.tok.pre/, then output the data in a dict {key: {words: count}}.
    """
    base_dir = os.path.join(DATA_DIR, "%d.tok.pre" % year)
    filelist = os.listdir(base_dir)
    tok = {}
    for filename in tqdm(filelist):
        key = os.path.splitext(filename)[0]
        with open(os.path.join(base_dir, filename), "r", encoding="UTF-8") as fp:
            text = fp.read().strip()
        tok[key] = text2dict(text)
    return tok


def load_all_tok(start_year: int, end_year: int) -> dict:
    """
    Load and merge all logvol files with year in [args.start_year, args.end_year].
    """
    tok = {}
    for year in range(start_year, end_year + 1):
        tok.update(load_tok(year))
    return tok


if __name__ == "__main__":
    pass
