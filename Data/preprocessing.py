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
WNL = WordNetLemmatizer()
SBS = SnowballStemmer("english")
STOP_WORDS = set(stopwords.words('english'))


def L_S_S(filename: str):
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

    with open(filename, "r", encoding="UTF-8") as fp:
        text = fp.read().strip()
    data = []
    tagged = pos_tag(word_tokenize(text))
    for tag in tagged:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        item = WNL.lemmatize(tag[0], pos=wordnet_pos)
        if (item in STOP_WORDS):
            continue
        if (("" == item) or ("\n" == item) or (" " == item)):
            raise
        item = SBS.stem(item)
        data.append(item)
    return ' '.join(data)


if __name__ == "__main__":
    pool = mul.Pool(mul.cpu_count())
    data = []
    for year in range(1996, 2007):
        base_dir = os.path.join(DATA_DIR, "%d.tok" % year)
        os.makedirs(base_dir + ".pre", exist_ok=True)
        filelist = os.listdir(base_dir)
        for filename in tqdm(filelist):
            data.append((os.path.join(base_dir + ".pre", filename), pool.apply_async(L_S_S, (os.path.join(base_dir, filename), ))))
    pool.close()

    for item in tqdm(data):
        with open(item[0], "w", encoding="UTF-8") as fp:
            print(item[1].get(), file=fp)
