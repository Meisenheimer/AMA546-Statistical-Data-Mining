import os
import time
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge

from DataLoader import load_all_logvol, load_all_tok, init_nltk
from TextAnalytics import get_vocabulary, calc_tf_idf
from Reduction import fit_reduction_model


def init(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    return None


def main(args: argparse.Namespace) -> None:
    # load data
    print("load data", file=args.log)
    logvol = load_all_logvol(args)
    tok = load_all_tok(args)
    keys = sorted(list(logvol.keys()))

    # compute tf-idf
    print("compute tf-idf", file=args.log)
    vocabulary = sorted(get_vocabulary(tok, keys, args))
    print("Size of vocabulary: %d." % len(vocabulary), file=args.log)
    tf_idf = calc_tf_idf(vocabulary, tok, keys, args)
    if (args.out_to_file):
        with open(os.path.join(args.output_dir, "Vocabulary.txt"), "w") as fp:
            for item in vocabulary:
                print(item, file=fp)

    # prepare data
    print("prepare data", file=args.log)
    index_train, index_test = train_test_split(list(range(len(keys))), test_size=args.test_size)

    if (len(set(index_train + index_test)) != len(keys)):
        raise

    train_x = tf_idf[index_train, :]
    test_x = tf_idf[index_test, :]
    train_y = np.zeros((len(index_train), 2))
    test_y = np.zeros((len(index_test), 2))
    for i in range(len(index_train)):
        train_y[i][0] = logvol[keys[index_train[i]]][0]
        train_y[i][1] = logvol[keys[index_train[i]]][1]
    for i in range(len(index_test)):
        test_y[i][0] = logvol[keys[index_test[i]]][0]
        test_y[i][1] = logvol[keys[index_test[i]]][1]

    if (args.use_residual):
        train_y[:, 1] = train_y[:, 1] - train_y[:, 0]
        train_y[:, 0] = 0.0
        test_y[:, 1] = test_y[:, 1] - test_y[:, 0]
        test_y[:, 0] = 0.0

    reduction_model = fit_reduction_model(train_x, train_y, args.reduct_method, args.target_dim)
    if (args.reduct_method == "NMF"):
        print(reduction_model.get_params(), file=args.log)
        print(reduction_model.components_, file=args.log)
    train_x = np.concatenate((train_y[:, 0].reshape(-1, 1), reduction_model.transform(train_x)), axis=1)
    test_x = np.concatenate((test_y[:, 0].reshape(-1, 1), reduction_model.transform(test_x)), axis=1)

    print("Size of train set: ", train_x.shape, train_y.shape, file=args.log)
    print("Size of test set: ", test_x.shape, test_y.shape, file=args.log)

    # training & results
    print("training & results", file=args.log)
    if (args.model == "Lasso"):
        model = Lasso(alpha=args.alpha, max_iter=args.iter)
    elif (args.model == "Ridge"):
        model = Ridge(alpha=args.alpha, max_iter=args.iter)
    else:
        raise
    model.fit(train_x, train_y[:, 1])
    print(model.coef_, file=args.log)
    print(model.intercept_, file=args.log)

    pred_y = model.predict(test_x)

    abs_error = np.abs(pred_y - test_y[:, 1])
    print(np.mean(abs_error), np.mean((abs_error) ** 2), np.max(abs_error), file=args.log)

    rel_error = np.abs((pred_y - test_y[:, 1]) / test_y[:, 1])
    print(np.mean(rel_error), np.max(rel_error), file=args.log)

    # baseline
    print("baseline", file=args.log)
    abs_error = np.abs(test_y[:, 0] - test_y[:, 1])
    print(np.mean(abs_error), np.mean((abs_error) ** 2), np.max(abs_error), file=args.log)

    rel_error = np.abs((test_y[:, 0] - test_y[:, 1]) / test_y[:, 1])
    print(np.mean(rel_error), np.max(rel_error), file=args.log)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="Lasso")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--reduct_method", type=str, default="None")
    parser.add_argument("--target_dim", type=int, default=-1)
    parser.add_argument("--start_year", type=int, default=1996)
    parser.add_argument("--end_year", type=int, default=2006)
    parser.add_argument("--use_residual", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--iter", type=int, default=16384)
    parser.add_argument("--del_stop_words", type=bool, default=False)
    parser.add_argument("--lemmatization", type=bool, default=False)
    parser.add_argument("--out_to_file", type=bool, default=False)

    args = parser.parse_args()
    args.time = time.localtime()

    if (args.out_to_file):
        args.output_dir = f"../Result/{args.model}_{args.alpha}_{args.reduct_method}_{args.target_dim}_{args.start_year}_{args.end_year}_{'_RES' if args.use_residual else ''}_{args.seed}_{args.time.tm_mon}-{args.time.tm_mday}-{args.time.tm_hour}-{args.time.tm_min}-{args.time.tm_sec}/"

        os.makedirs("../Result/", exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)

        args.log = open(os.path.join(args.output_dir, "log"), "w", encoding="UTF-8")
    else:
        args.log = None

    print(args, file=args.log)

    init_nltk()

    init(args)
    main(args)

    if (args.out_to_file):
        args.log.close()
