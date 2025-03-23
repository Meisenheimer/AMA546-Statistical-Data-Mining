import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import NMF

from DataLoader import load_all_logvol, load_all_tok, load_logvol, load_tok
from TextAnalytics import get_vocabulary, calc_tf_idf


def init(args):
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    return None


def error(pred_y: np.ndarray, true_y: np.ndarray) -> tuple:
    """
    Compute the MAE and MSE for given result with the ground truth.
    """
    abs_error = np.abs(pred_y - true_y)
    return (np.mean(abs_error), np.mean(abs_error ** 2))


def train_test(args: argparse.Namespace,
               train_x: np.ndarray, train_y: np.ndarray,
               test_x: np.ndarray, test_y: np.ndarray,
               H: np.ndarray) -> tuple:
    """
    Train the model on the given data, then output the MAE, MSE (compare with Naive model) and the weights of words.
    """
    if (args.model == "Lasso"):
        model = Lasso(alpha=args.alpha, max_iter=args.iter)
    elif (args.model == "Ridge"):
        model = Ridge(alpha=args.alpha, max_iter=args.iter)
    elif (args.model == "DecisionTree"):
        model = DecisionTreeRegressor(max_depth=None if args.max_depth == 0 else args.max_depth, min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf)
    else:
        raise

    model.fit(train_x, train_y[:, 1])

    if (args.model in ["Lasso", "Ridge"]):
        coef = model.coef_[1:].reshape(args.target_dim)
        print(model.coef_, file=args.log)
        print(model.intercept_, file=args.log)
    elif (args.model == "DecisionTree"):
        coef = model.feature_importances_[1:].reshape(args.target_dim)
        print(model.feature_importances_, file=args.log)
    else:
        pass

    word_weight = np.linalg.pinv(H) @ coef

    pred_y = model.predict(test_x)

    return (error(pred_y, test_y[:, 1]), error(test_y[:, 0], test_y[:, 1]), word_weight)


def main(args: argparse.Namespace) -> None:
    """
    The algorithm is shown in the slide and report.
    """

    # load the data from text file.
    print("Loading data.")
    logvol = load_all_logvol(args.start_year, args.end_year)
    tok = load_all_tok(args.start_year, args.end_year)
    keys = sorted(list(logvol.keys()))

    # calculate the vocabulary set and TF-IDF.
    vocabulary = sorted(get_vocabulary(tok, keys, args))
    print("Size of vocabulary: %d." % len(vocabulary))
    tf_idf = calc_tf_idf(vocabulary, tok, keys, args)

    # Use NMF for dimensionality reduction.
    print("Computing NMF.")
    NMFModel = NMF(n_components=args.target_dim).fit(tf_idf)
    for_norm = np.linalg.norm(tf_idf, "fro")
    tf_idf = NMFModel.transform(tf_idf)
    H = NMFModel.components_
    e = (tf_idf @ H)
    print("NMF Non-zero rate = %f%%; NMF Error = %f/%f." % (np.count_nonzero(H) / float(H.shape[0] * H.shape[1]), np.linalg.norm(e, "fro"), for_norm), file=args.log)

    # Prepare the training data.
    y = np.zeros((len(keys), 2))
    for i in range(len(keys)):
        y[i, 0] = logvol[keys[i]][0]
        y[i, 1] = logvol[keys[i]][1]
    tf_idf = np.concatenate((y[:, 0].reshape(-1, 1), tf_idf), axis=1)\

    mae = []
    mae_naive = []
    mse = []
    mse_naive = []
    word_weights = np.zeros(len(vocabulary))

    print("Training and testing.")
    for seed in tqdm(range(args.epoch)):
        # Train and test the data with different splitting, and then take the average as the result.
        args.seed = seed
        init(args)

        train_x, test_x, train_y, test_y = train_test_split(tf_idf, y, test_size=args.test_size)

        res, res_naive, word_weight = train_test(args, train_x, train_y, test_x, test_y, H)

        mae.append((res[0]))
        mae_naive.append(res_naive[0])
        mse.append((res[1]))
        mse_naive.append(res_naive[1])
        word_weights += word_weight

    print("Mean MAE = %f/%f, Var MAE = %f/%f." % (np.mean(mae), np.mean(mae_naive), np.var(mae), np.var(mae_naive)), file=args.log)
    print("Mean MSE = %f/%f, Var MSE = %f/%f." % (np.mean(mse), np.mean(mse_naive), np.var(mse), np.var(mse_naive)), file=args.log)
    word_weights /= float(args.epoch)

    with open(os.path.join(args.output_dir, "MAE.txt"), "w", encoding="UTF-8") as fp:
        for i in range(len(mae)):
            print(mae[i], mae_naive[i], file=fp)

    with open(os.path.join(args.output_dir, "MSE.txt"), "w", encoding="UTF-8") as fp:
        for i in range(len(mse)):
            print(mse[i], mse_naive[i], file=fp)

    # Use the previous data to train the model, and test on the new year's data.
    print("Predict new data (year = %d)." % (args.end_year + 1))
    new_logvol = load_logvol(args.end_year + 1)
    new_tok = load_tok(args.end_year + 1)
    new_keys = sorted(list(new_logvol.keys()))
    new_tf_idf = calc_tf_idf(vocabulary, new_tok, new_keys, args)  # use the vocabulary of the previous to compute the TF-IDF of the new data.
    new_tf_idf = NMFModel.transform(new_tf_idf)

    ny = np.zeros((len(new_keys), 2))
    for i in range(len(new_keys)):
        ny[i, 0] = new_logvol[new_keys[i]][0]
        ny[i, 1] = new_logvol[new_keys[i]][1]
    nx = np.concatenate((ny[:, 0].reshape(-1, 1), new_tf_idf), axis=1)

    new_res, new_res_naive, new_word_weight = train_test(args, tf_idf, y, nx, ny, H)

    print("Predict: MAE = %f/%f." % (new_res[0], new_res_naive[0]), file=args.log)
    print("Predict: MSE = %f/%f." % (new_res[1], new_res_naive[1]), file=args.log)

    with open(os.path.join(args.output_dir, "Vocabulary.txt"), "w") as fp:
        for i in range(len(vocabulary)):
            print(vocabulary[i], new_word_weight[i], word_weights[i], file=fp)
    return None


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="Lasso")

    parser.add_argument("--alpha", type=float, default=1.0)  # For lasso and ridge.

    parser.add_argument("--max_depth", type=int, default=5)  # For decision tree.
    parser.add_argument("--min_samples_split", type=int, default=2)  # For decision tree.
    parser.add_argument("--min_samples_leaf", type=int, default=1)  # For decision tree.

    parser.add_argument("--target_dim", type=int, default=20)  # For NMF.

    parser.add_argument("--start_year", type=int, default=1996)
    parser.add_argument("--end_year", type=int, default=1996)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--iter", type=int, default=16384)

    args = parser.parse_args()
    args.time = time.localtime()

    args.output_dir = f"../Result/{args.model}_{args.start_year}_{args.end_year}_{args.time.tm_mon:02d}{args.time.tm_mday:02d}-{args.time.tm_hour:02d}{args.time.tm_min:02d}{args.time.tm_sec:02d}/"

    os.makedirs("../Result/", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    args.log = open(os.path.join(args.output_dir, "log.txt"), "w", encoding="UTF-8")

    print(args, file=args.log)

    init(args)
    main(args)

    args.log.close()

    print("Total time = %f(s)" % (time.time() - start_time))
