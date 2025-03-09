import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge

from DataLoader import load_all_logvol, load_all_tok
from TextAnalytics import get_vocabulary, calc_tf_idf


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
    vocabulary = get_vocabulary(tok, keys)
    tf, idf, tf_idf = calc_tf_idf(vocabulary, tok, keys)

    # prepare data
    Y = np.zeros((len(keys), 2))
    for i in range(len(keys)):
        Y[i][0] = logvol[keys[i]][0]
        Y[i][1] = logvol[keys[i]][1]
    X = np.concatenate((Y[:, 0].reshape(-1, 1), tf_idf), axis=1)
    # X = Y[:, 0].reshape(-1, 1)

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=args.test_size)

    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    print(test_y.shape)

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
    parser.add_argument("--iter", type=int, default=2048)

    parser.add_argument("--warm_start", type=bool, default=False)
    parser.add_argument("--model", type=str, default="Lasso")

    args = parser.parse_args()

    print(args)

    init(args)
    main(args)
