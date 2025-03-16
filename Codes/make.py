target_dim = 20
alpha = 0.0001
use_residual = False
with open("run.sh", "w") as fp:
    for start_year in range(1996, 2006):
        print(f'python main.py --model "Lasso" --alpha {alpha} --reduct_method "NMF" --target_dim {target_dim} --start_year {start_year} --end_year {start_year}{" --use_residual True" if use_residual else ""} --del_stop_words True --lemmatization True --stemming True --use_pre_data True', file=fp)

        print(f'python main.py --model "Ridge" --alpha {alpha} --reduct_method "NMF" --target_dim {target_dim} --start_year {start_year} --end_year {start_year}{" --use_residual True" if use_residual else ""} --del_stop_words True --lemmatization True --stemming True --use_pre_data True', file=fp)

        print(f'python main.py --model "DecisionTree" --reduct_method "NMF" --target_dim {target_dim} --start_year {start_year} --end_year {start_year}{" --use_residual True" if use_residual else ""} --del_stop_words True --lemmatization True --stemming True --use_pre_data True', file=fp)
