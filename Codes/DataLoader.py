import os
import argparse
from tqdm import tqdm

DATA_DIR = "../Data/Data/"


def load_logvol(year: int) -> dict:
    """
    load {year}.logvol.-12.txt and {year}.logvol.+12.txt, then output the data in a dict {key: (preceding year, following year)}.
    """
    fp = open(os.path.join(DATA_DIR, "%d.logvol.-12.txt" % year), "r")
    text_1 = fp.readlines()
    fp.close()
    fp = open(os.path.join(DATA_DIR, "%d.logvol.+12.txt" % year), "r")
    text_2 = fp.readlines()
    fp.close()
    data = {}
    if (len(text_1) != len(text_2)):
        raise
    for i in range(len(text_1)):
        tmp_1 = text_1[i].replace("\n", "").split(' ')
        tmp_2 = text_2[i].replace("\n", "").split(' ')
        if (len(tmp_1) != 2 or len(tmp_2) != 2):
            raise
        if (tmp_1[1] != tmp_2[1]):
            raise
        data[tmp_1[1]] = (float(tmp_1[0]), float(tmp_2[0]))
    return data


def load_tok(year: int) -> dict:
    """
    load all files under {year}.tok/, then output the data in a dict {key: list of words}
    """
    base_dir = os.path.join(DATA_DIR, "%d.tok" % year)
    filelist = os.listdir(base_dir)
    data = {}
    for filename in filelist:
        fp = open(os.path.join(base_dir, filename), "r")
        text = fp.read()
        fp.close()
        text = text.replace("\n", "").split(' ')
        while ("" in text):
            text.remove("")
        data[os.path.splitext(filename)[0]] = text
    return data


def load_all_logvol(args: argparse.Namespace) -> dict:
    """
    load and merge all logvol files from args.start_year to args.end_year
    """
    cnt = 0
    logvol = {}
    for year in tqdm(range(args.start_year, args.end_year + 1)):
        tmp = load_logvol(year)
        cnt += len(tmp)
        logvol.update(tmp)
    if (cnt != len(logvol)):
        raise
    return logvol


def load_all_tok(args: argparse.Namespace) -> dict:
    """
    load and merge all tok files from args.start_year to args.end_year
    """
    cnt = 0
    tok = {}
    for year in tqdm(range(args.start_year, args.end_year + 1)):
        tmp = load_tok(year)
        cnt += len(tmp)
        tok.update(tmp)
    if (cnt != len(tok)):
        raise
    return tok


if __name__ == "__main__":
    pass
