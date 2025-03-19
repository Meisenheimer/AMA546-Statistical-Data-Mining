with open("run.sh", "w") as fp:
    for start_year in range(1996, 2006):
        print(f'python main.py --model "Lasso" --alpha 0.00005 --reduct_method "NMF" --target_dim 20 --start_year {start_year} --end_year {start_year} --del_stop_words True --lemmatization True --stemming True --use_pre_data True --use_residual True', file=fp)
        if (start_year <= 2003):
            print(f'python main.py --model "Lasso" --alpha 0.00005 --reduct_method "NMF" --target_dim 20 --start_year {start_year} --end_year {start_year + 2} --del_stop_words True --lemmatization True --stemming True --use_pre_data True --use_residual True', file=fp)

    for start_year in range(1996, 2006):
        print(f'python main.py --model "Ridge" --alpha 1.0 --reduct_method "NMF" --target_dim 20 --start_year {start_year} --end_year {start_year} --del_stop_words True --lemmatization True --stemming True --use_pre_data True --use_residual True', file=fp)
        if (start_year <= 2003):
            print(f'python main.py --model "Ridge" --alpha 1.0 --reduct_method "NMF" --target_dim 20 --start_year {start_year} --end_year {start_year + 2} --del_stop_words True --lemmatization True --stemming True --use_pre_data True --use_residual True', file=fp)

    for start_year in range(1996, 2006):
        print(f'python main.py --model "DecisionTree" --max_depth 0 --min_samples_split 512 --min_samples_leaf 256 --reduct_method "NMF" --target_dim 20 --start_year {start_year} --end_year {start_year} --del_stop_words True --lemmatization True --stemming True --use_pre_data True --use_residual True', file=fp)
        if (start_year <= 2003):
            print(f'python main.py --model "DecisionTree" --max_depth 0 --min_samples_split 512 --min_samples_leaf 256 --reduct_method "NMF" --target_dim 20 --start_year {start_year} --end_year {start_year + 2} --del_stop_words True --lemmatization True --stemming True --use_pre_data True --use_residual True', file=fp)
