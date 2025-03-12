with open("run.sh", "w") as fp:
    for target_dim in [10, 100]:
        for use_residual in [True, False]:
            for model in ["Lasso", "Ridge"]:
                for alpha in [0.01, 0.005, 0.001]:
                    for reduct_model in ["NMF"]:
                        for start_year in [1996]:
                            for end_year in [2001]:
                                for seed in [40, 42, 44]:
                                    print(f"python main.py --model {model} --alpha {alpha} --reduct_method {reduct_model} --target_dim {target_dim} --start_year {start_year} --end_year {end_year}{' --use_residual True' if use_residual else ''} --seed {seed} --del_stop_words True", file=fp)
