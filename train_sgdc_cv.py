from preprocess import PreProcess
import numpy as np
import logging
import csv
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV

if __name__ == "__main__":
    logging.basicConfig(format='--%(asctime)s:[%(levelname)s]:%(lineno)d:%(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
    logging.info("train_sgdc_cv.py Start")
    p = PreProcess()

    out_filepath = 'data/train_1M.csv.out'
    #out_filepath = 'data/train_s404_100K.out.1vs1'
    #out_filepath = 'data/train_1000.csv.out'

    #Load data
    X, y = p.load_train_data(out_filepath)
    logging.info("Shape X = %r, y =%r" %(X.shape, y.shape ))
    logging.info("example X = %s\ny =%r" %(X[0], y[0]))
    logging.info("classes: %r" % list(np.unique(y)))

    #Sampling
    #At least 3
    POWER = 6
    CONST = 1
    #At least 2
    #CV = 5
    n_subsamples = CONST*10**POWER
    n_size = y.shape[0]
    if n_subsamples < n_size:
        logging.info("second index : %d to %d" % (n_size - n_subsamples/2, n_size))
        X_small_train = X[range(n_subsamples/2)+range(n_size - n_subsamples/2, n_size)]
        y_small_train = y[range(n_subsamples/2)+range(n_size - n_subsamples/2, n_size)]
    else:
        X_small_train, y_small_train = X, y
    n_subsamples = y_small_train.shape[0]
    logging.info("Samples : %d" % (n_subsamples))
    
    clf_params = {
            #'alpha': [0.0001],
            'alpha': np.logspace(-5, -1, 4),
            #'n_iter': [5],
            'n_iter': [50, 80, 110, 140],
            'penalty': ['l1','l2', 'elasticnet']
            #'penalty': ['l2']
            }
    #Add class_weight
    class_weight = {
            1:4.8,
            0:1.0
            }
    clf = linear_model.SGDClassifier(loss='log', n_jobs=-1, class_weight='balanced', shuffle=True, verbose = 0)

    gs_clf = GridSearchCV(clf, clf_params, refit=True ,scoring='log_loss', n_jobs=-1, verbose = 1)

    gs_clf.fit(X_small_train, y_small_train)

    logging.info("Best params: %s\nScore: %s" % (gs_clf.best_params_, gs_clf.best_score_))
    
    #XXX skip test
    exit()

    #Load test data for 5 times
    test_filepattern = 'data/test_%d_M.out'
    field = ['id', 'click']
    for part in range(1, 6):
        test_filepath = test_filepattern % part
        logging.info("Loading test set [%s]..." % test_filepath)
        X_test, ids_test= p.load_test_data(test_filepath)
        logging.info("Shape X = %r, ids =%r" %(X_test.shape, ids_test.shape ))
        logging.info("example X = %s\nids =%r" %(X_test[0], ids_test[0]))
        svc_probs = gs_clf.predict_proba(X_test)
        #[prob of 0, prob of 1]
        logging.info("prob of test: %s" % svc_probs[:10])
        
        out_filepath = "%s-sgdc-s%d-a%f-p%s-i%d.csv" %(test_filepath, n_subsamples, gs_clf.best_params_['alpha'], gs_clf.best_params_['penalty'], gs_clf.best_params_['n_iter'])
        logging.info("Writing out file %s" % out_filepath)
        if len(ids_test) != len(svc_probs):
            logging.error("Test case count don:t match")
        else :
            with open(out_filepath, 'a') as ofile:
                writer = csv.DictWriter(ofile, field)
                if part == 1:
                    writer.writeheader()
                for i in range(len(ids_test)):
                    row = {'id' : ids_test[i], 'click' : svc_probs[i][1]}
                    #logging.info("row %d : %s" %(i, row))
                    writer.writerow(row)

    
    logging.info("train_sgdc_cv.py End")
