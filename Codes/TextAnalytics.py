from tqdm import tqdm
import numpy as np


def get_vocabulary(words_dict: dict, keys: list) -> dict:
    """
    compute the vocabulary for tokens
    """
    vocabulary = {}
    num = 0
    for key in tqdm(keys):
        for item in words_dict[key]:
            if (item not in vocabulary.keys()):
                vocabulary[item] = num
                num += 1
    if (len(vocabulary) != num):
        raise
    return vocabulary


def calc_tf_idf(vocabulary: dict, words_dict: dict, keys: list) -> tuple:
    """
    compute the tf_idf for tokens
    """
    tf = np.zeros((len(words_dict), len(vocabulary)))
    keys = list(keys)
    for i in tqdm(range(len(keys))):
        key = keys[i]
        for item in words_dict[key]:
            tf[i, vocabulary[item]] += 1.0
    idf = np.log2(float(tf.shape[0]) / np.count_nonzero(tf, axis=0))
    tf_idf = tf * idf
    return tf, idf, tf_idf


if __name__ == "__main__":
    pass
