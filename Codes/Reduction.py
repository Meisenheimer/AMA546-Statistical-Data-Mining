import numpy as np
from sklearn.decomposition import PCA
from typing import Union
from sklearn.decomposition import NMF


class dr_trival:
    def transform(self, data: np.ndarray) -> np.ndarray:
        return data


def dr_pca(x: np.ndarray, y: Union[np.ndarray, None] = None, target_dim: int = 0) -> np.ndarray:
    return PCA(n_components=target_dim if target_dim > 0 else "mle").fit(x, y)


def dr_nmf(x: np.ndarray, y: Union[np.ndarray, None] = None, target_dim: int = 0) -> np.ndarray:
    return NMF(n_components=target_dim).fit(x, y)


def fit_reduction_model(x: np.ndarray, y: Union[np.ndarray, None] = None,
                        dim_reduct_method: str = "None", target_dim: int = 0) -> object:
    if (dim_reduct_method != "None"):
        print("Use %s for reduction with %d components." % (dim_reduct_method, target_dim))
    if (dim_reduct_method == "None"):
        return dr_trival()
    elif (dim_reduct_method.upper() == "PCA"):
        return dr_pca(x, y, target_dim)
    elif (dim_reduct_method.upper() == "NMF"):
        return dr_nmf(x, y, target_dim)
    raise


if __name__ == "__main__":
    pass
