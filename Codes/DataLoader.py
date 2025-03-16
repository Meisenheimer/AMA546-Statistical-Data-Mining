import os
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import argparse
from tqdm import tqdm
import multiprocessing as mul

DATA_DIR = "../Data/Data/"
PRE_DATA_DIR = "../Data/Preprocessed/"

nltk.data.path.append("./nltk/")
STOP_WORDS = set(stopwords.words('english'))
WNL = WordNetLemmatizer()
SBS = SnowballStemmer("english")


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


def text2dict(text: str, lemmatization: bool, del_stop_words: bool, stemming: bool):
    data = {}
    if (lemmatization):
        def get_wordnet_pos(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return None
        tagged = pos_tag(word_tokenize(text))
        for tag in tagged:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            item = WNL.lemmatize(tag[0], pos=wordnet_pos)
            if (del_stop_words and (item in STOP_WORDS)):
                continue
            if (("" == item) or ("\n" == item) or (" " == item)):
                raise
            if (stemming):
                item = SBS.stem(item)
            if (item not in data):
                data[item] = 0
            data[item] += 1
    else:
        for item in text.split(' '):
            if (del_stop_words and (item in STOP_WORDS)):
                continue
            if (("" == item) or ("\n" == item) or (" " == item)):
                raise
            if (stemming):
                item = SBS.stem(item)
            if (item not in data):
                data[item] = 0
            data[item] += 1
    return data


def load_tok(year: int, args: argparse.Namespace) -> dict:
    """
    load all files under {year}.tok/, then output the data in a dict {key: list of words}
    """
    if (args.use_pre_data):
        base_dir = os.path.join(DATA_DIR, "%d.tok.pre" % year)
    else:
        base_dir = os.path.join(DATA_DIR, "%d.tok" % year)
    filelist = os.listdir(base_dir)
    tok = {}
    if (not args.use_pre_data):
        pool = mul.Pool(mul.cpu_count())
    for filename in tqdm(filelist):
        key = os.path.splitext(filename)[0]
        with open(os.path.join(base_dir, filename), "r", encoding="UTF-8") as fp:
            text = fp.read().strip()
        if (not args.use_pre_data):
            tok[key] = pool.apply_async(text2dict, (text, args.lemmatization, args.del_stop_words, args.stemming))
        else:
            tok[key] = text2dict(text, args.lemmatization, args.del_stop_words, args.stemming)
    if (not args.use_pre_data):
        pool.close()
        for filename in tqdm(filelist):
            key = os.path.splitext(filename)[0]
            tok[key] = tok[key].get(None)
    return tok


def load_all_tok(args: argparse.Namespace) -> dict:
    cnt = 0
    tok = {}

    for year in range(args.start_year, args.end_year + 1):
        tmp = load_tok(year, args)
        cnt += len(tmp)
        tok.update(tmp)
    if (cnt != len(tok)):
        raise
    return tok


if __name__ == "__main__":
    pass
