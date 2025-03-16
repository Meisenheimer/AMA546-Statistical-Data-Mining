import argparse
import numpy as np
from tqdm import tqdm


def get_vocabulary(words_dict: dict, keys: list, args: argparse.Namespace) -> list:
    """
    compute the vocabulary for tokens
    """
    vocabulary = set()
    num = 0
    for key in tqdm(keys):
        for item in words_dict[key].keys():
            if (item not in vocabulary):
                vocabulary.add(item)
    return list(vocabulary)


def calc_tf_idf(vocabulary: list, words_dict: dict, keys: list, args: argparse.Namespace) -> tuple:
    """
    compute the tf, idf, tf_idf for tokens
    """
    n_key = len(keys)
    n_vocabulary = len(vocabulary)
    vocabulary_map = {}
    k = 0
    for item in vocabulary:
        vocabulary_map[item] = k
        k += 1
    if (n_key != len(words_dict)):
        raise
    tf = np.zeros((n_key, n_vocabulary))
    for i in tqdm(range(n_key)):
        tmp = words_dict[keys[i]]
        key_set = set(tmp.keys()) & set(vocabulary)
        for item in key_set:
            tf[i, vocabulary_map[item]] = tmp[item]
    tf /= tf.sum(axis=1).reshape(-1, 1)
    tf *= np.log2(float(tf.shape[0]) / np.clip(np.count_nonzero(tf, axis=0), a_min=1, a_max=None))
    return tf


if __name__ == "__main__":
    pass
