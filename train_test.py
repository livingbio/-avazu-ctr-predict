from preprocess import PreProcess
import numpy as np
import logging
from sklearn.svm import SVC

if __name__ == "__main__":
    logging.basicConfig(format='--%(asctime)s:[%(levelname)s]:%(lineno)d:%(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
    logging.info("train_test.py Start")
    p = PreProcess()

    train_filepath = 'data/train_1000.csv.out'
    test_filepath = 'data/test_1000.csv.out'

    #Load train data
    logging.info("Loading train set...")
    X_train, y_train= p.load_train_data(train_filepath)
    logging.info("Shape X_train = %r, y_train =%r" %(X_train.shape, y_train.shape ))
    logging.info("example X_train = %s\ny_train =%r" %(X_train[0], y_train[0]))
    logging.info("classes: %r" % list(np.unique(y_train)))
    
    #Training
    C = 0.10000000000000001
    gamma = 0.0001 
    logging.info("Train SVC with C: %d, gamma: %d" %(C, gamma))
    svc = SVC(C=C, gamma = gamma, probability=True, verbose = True).fit(X_train, y_train)
    
    #Load test data
    logging.info("Loading test set...")
    X_test, ids_test= p.load_test_data(test_filepath)
    logging.info("Shape X = %r, ids =%r" %(X_test.shape, ids_test.shape ))
    logging.info("example X = %s\nids =%r" %(X_test[0], ids_test[0]))
    svc_probs = svc.predict_proba(X_test)
    #TODO make result file
    #[prob of 0, prob of 1]
    logging.info(svc_probs[:10])
    
    logging.info("train_test.py End")
