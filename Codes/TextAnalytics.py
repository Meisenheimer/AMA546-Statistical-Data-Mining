import argparse
import numpy as np
from tqdm import tqdm


def get_vocabulary(words_dict: dict, keys: list, args: argparse.Namespace) -> list:
    """
    compute the vocabulary for tokens
    """
    vocabulary = set()
    num = 0
    for key in tqdm(keys, file=args.log):
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
    for i in tqdm(range(n_key), file=args.log):
        tmp = words_dict[keys[i]]
        for item in tmp.keys():
            tf[i, vocabulary_map[item]] = tmp[item]
    tf *= np.log2(float(tf.shape[0]) / np.count_nonzero(tf, axis=0))
    return tf


if __name__ == "__main__":
    pass
