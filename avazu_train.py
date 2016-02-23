from preprocess import PreProcess
import numpy as np
import logging
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

if __name__ == "__main__":
    logging.basicConfig(format='--%(asctime)s:[%(levelname)s]:%(lineno)d:%(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
    p = PreProcess()
    PREPROCESS = False

    filepath = 'data/train_1M.csv'
    out_filepath = 'data/train_1M.csv.out'
    #out_filepath = 'data/train_1000.csv.out'
    if PREPROCESS:
        out_filepath = p.convert(filepath)

    #Load data
    X, y = p.load_train_data(out_filepath)
    logging.info("Shape X = %r, y =%r" %(X.shape, y.shape ))
    logging.info("example X = %s\ny =%r" %(X[0], y[0]))
    logging.info("classes: %r" % list(np.unique(y)))

    #CV
    svc_params = {
            'C': np.logspace(-1, 2, 4),
            'gamma': np.logspace(-4, 0, 5),
            }

    #Sampling
    POWER = 6
    CONST = 1
    CV = 5
    n_subsamples = CONST*10**POWER
    n_size = y.shape[0]
    if n_subsamples < n_size:
        X_small_train, y_small_train = X[:n_subsamples], y[:n_subsamples]
    else:
        X_small_train, y_small_train = X, y
    logging.info("Samples : %d, CV :%d" % (n_size, CV))
    
    gs_svc = GridSearchCV(SVC(), svc_params, cv=CV, n_jobs=-1)

    gs_svc.fit(X_small_train, y_small_train)

    logging.info("Best params: %s\nScore: %s" % (gs_svc.best_params_, gs_svc.best_score_)) # {'C': 10.0, 'gamma': 0.001} 0.982

