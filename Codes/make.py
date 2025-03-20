with open("run.sh", "w") as fp:
    for start_year in range(1996, 2006):
        print(f'python main.py --model "Lasso" --alpha 0.00005 --target_dim 20 --start_year {start_year} --end_year {start_year}', file=fp)
        if (start_year <= 2003):
            print(f'python main.py --model "Lasso" --alpha 0.00005 --target_dim 20 --start_year {start_year} --end_year {start_year + 2}', file=fp)

    for start_year in range(1996, 2006):
        print(f'python main.py --model "Ridge" --alpha 1.0 --target_dim 20 --start_year {start_year} --end_year {start_year}', file=fp)
        if (start_year <= 2003):
            print(f'python main.py --model "Ridge" --alpha 1.0 --target_dim 20 --start_year {start_year} --end_year {start_year + 2}', file=fp)

    for start_year in range(1996, 2006):
        print(f'python main.py --model "DecisionTree" --max_depth 5 --target_dim 20 --start_year {start_year} --end_year {start_year} --use_residual True', file=fp)
        if (start_year <= 2003):
            print(f'python main.py --model "DecisionTree" --max_depth 5 --target_dim 20 --start_year {start_year} --end_year {start_year + 2} --use_residual True', file=fp)
