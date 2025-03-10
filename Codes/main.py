import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge

from DataLoader import load_all_logvol, load_all_tok
from TextAnalytics import get_vocabulary, calc_tf_idf
from Reduction import fit_reduction_model


def init(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    return None


def main(args: argparse.Namespace) -> None:
    # load data
    logvol = load_all_logvol(args)
    tok = load_all_tok(args)
    keys = list(logvol.keys())

    # compute tf-idf
    vocabulary = get_vocabulary(tok, keys, args.del_stop_words)
    tf, idf, tf_idf = calc_tf_idf(vocabulary, tok, keys)
    print("Size of vocabulary: %d." % len(vocabulary))
    with open("../Result/Vocabulary.txt", "w") as fp:
        for item in vocabulary:
            print(item, file=fp)

    # prepare data
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

    reduction_model = fit_reduction_model(train_x, train_y if args.use_train_y else None,
                                          args.reduct_method, args.target_dim)
    train_x = np.concatenate((train_y[:, 0].reshape(-1, 1), reduction_model.transform(train_x)), axis=1)
    test_x = np.concatenate((test_y[:, 0].reshape(-1, 1), reduction_model.transform(test_x)), axis=1)

    print("Size of train set: ", train_x.shape, train_y.shape)
    print("Size of test set: ", test_x.shape, test_y.shape)

    # training & results
    if (args.model == "Lasso"):
        model = Lasso(alpha=args.alpha, max_iter=args.iter, warm_start=args.warm_start)
        if (args.warm_start):
            init_coef = np.zeros(X.shape[1])
            init_coef[0] = 1.0
            model.coef_ = init_coef
            model.intercept_ = 0.0
    elif (args.model == "Ridge"):
        model = Ridge(alpha=args.alpha, max_iter=args.iter)
    else:
        raise
    model.fit(train_x, train_y[:, 1])
    print(model.coef_)
    print(model.intercept_)

    pred_y = model.predict(test_x)

    abs_error = np.abs(pred_y - test_y[:, 1])
    print(np.mean(abs_error), np.mean((abs_error) ** 2), np.max(abs_error))

    rel_error = np.abs((pred_y - test_y[:, 1]) / test_y[:, 1])
    print(np.mean(rel_error), np.max(rel_error))

    # baseline
    abs_error = np.abs(test_y[:, 0] - test_y[:, 1])
    print(np.mean(abs_error), np.mean((abs_error) ** 2), np.max(abs_error))

    rel_error = np.abs((test_y[:, 0] - test_y[:, 1]) / test_y[:, 1])
    print(np.mean(rel_error), np.max(rel_error))

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_size", type=float, default=0.25)

    parser.add_argument("--start_year", type=int, default=1996)
    parser.add_argument("--end_year", type=int, default=2006)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--iter", type=int, default=8192)

    parser.add_argument("--warm_start", type=bool, default=False)
    parser.add_argument("--model", type=str, default="Lasso")

    parser.add_argument("--del_stop_words", type=bool, default=False)

    parser.add_argument("--reduct_method", type=str, default="None")
    parser.add_argument("--target_dim", type=int, default=-1)
    parser.add_argument("--use_train_y", type=bool, default=False)

    args = parser.parse_args()

    print(args)

    init(args)
    main(args)
