import os
import argparse
from tqdm import tqdm

DATA_DIR = "../Data/Data/"
PRE_DATA_DIR = "../Data/Preprocessed/"


def load_logvol(year: int) -> dict:
    """
    load {year}.logvol.-12.txt and {year}.logvol.+12.txt, then output the data in a dict {key: (preceding year, following year)}.
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
        with open(os.path.join(base_dir, filename), "r", encoding="UTF-8") as fp:
            text = fp.read()
        text = text.strip().split(' ')
        if ("" in text or "\n" in text or " " in text):
            raise
        data[os.path.splitext(filename)[0]] = text
    return data


def load_all_logvol(args: argparse.Namespace) -> dict:
    """
    load and merge all logvol files from args.start_year to args.end_year
    """
    cnt = 0
    logvol = {}
    for year in tqdm(range(args.start_year, args.end_year + 1), file=args.log):
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
    for year in tqdm(range(args.start_year, args.end_year + 1), file=args.log):
        tmp = load_tok(year)
        cnt += len(tmp)
        tok.update(tmp)
    if (cnt != len(tok)):
        raise
    return tok


def load_all_tok_dict(args: argparse.Namespace) -> dict:
    tok = {}
    for year in tqdm(range(args.start_year, args.end_year + 1), file=args.log):
        base_dir = os.path.join(PRE_DATA_DIR, "%d.tok" % year)
        filelist = os.listdir(base_dir)
        for filename in filelist:
            key = os.path.splitext(filename)[0]
            tok[key] = {}
            with open(os.path.join(base_dir, filename), "r", encoding="UTF-8") as fp:
                text = fp.readlines()
            for line in text:
                tmp = line.split(' ')
                tok[key][tmp[0]] = int(tmp[1])
    return tok


if __name__ == "__main__":
    pass
