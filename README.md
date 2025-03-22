# Statistical-Data-Mining-Project

**If there is any problem, please email me at [zeyu-asparagine.wang@connect.polyu.hk](mailto:zeyu-asparagine.wang@connect.polyu.hk) .**

Requirement:

- `Python`: the required pacckages are listed in `requirements.txt`, most are commonly used (e.g. scikit-learn, numpy, ...), except nltk.

The followings are description for each folder:

- `./Codes/`: The main codes for the project, except the lemmatization, stemming and stop words filter part;
- `./Data/`: The codes for downloading data, lemmatizing, stemming and stop words filtering;
- `./Report/`: The final report (**Some result in the report is different from the slide cause we  choose some better parameters**);
- `./Result/`: Some results to support our report and slide. The parameters will be set as shown in the report;
- `./Slide/`: The slide for presentation;

## Preprocessing (`./Data/`)

1. Downloading the data from the website by running `python download_data.py`, all the data needed will be download to `./Origin/` and unzip to `./Data/`;
2. Preprocessing the data (including lemmatizing, stemming and stop words filtering) by running `python preprocessing.py`, the files in `./Data/{year}.tok/` will be load and processed and output to `./Data/{year}.tok.pre/`.

**This process will take much time, even though the multiprocessing is used. Thus it is not necessary unless you want to check whether the nltk package always gives the same resutls.**

## Main part (`./Codes/`)

There is a `run.sh` in this folder, which including the testing shown as report. You can simply run as it or try more test with the command:

```bash
python --model [Model for testing {Lasso, Ridge, DecisionTree}] --alpha [Parameter for Lasso/Ridge] --max_depth [Parameter for DecisionTree] --min_samples_split [Parameter for DecisionTree] --min_samples_leaf [Parameter for DecisionTree] --target_dim [The number of component given by NMF] --start_year [The time interval for input data (included this year)] --end_year [The time interval for input data (included this year)]
```

There are some other parameters like seed, epoch (the time for repeating the experiment), test_size (rate of the testing data) and iter (the maximum iteration for model fitting), which are not needed to adjust unless any problem occurs e.g. the model is not convergent.

## Result (`./Result/`)

The `draw.py` will draw the MAE and MSE for the result, and `words.py` will analysis the common part and difference of the words features given by different models and through different years.

The result will be shown on the report.

The codes will load the data from `./Res-1` and `./Res-3`, if you want to check the correctness, you need to delete the results in two folders and move the new result in (the `./Res-1/` for 1-year data and `./Res-3` for 3 years). Or you can simply check the MAE and MSE at the end of `log.txt`, and the Mean MAE/MSE shows the testing error while the Predict: MAE/MSE shows the predict error.