from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
import scipy
import numpy as np

mem = Memory("./mycache")

@mem.cache
def get_data(f):
    data = load_svmlight_file(f)
    return scipy.sparse.csr_matrix.toarray(data[0]), data[1].astype(np.int64)

