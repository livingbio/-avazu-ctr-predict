from preprocess import PreProcess
import numpy as np
import logging
import csv
from sklearn.svm import SVC

if __name__ == "__main__":
    logging.basicConfig(format='--%(asctime)s:[%(levelname)s]:%(lineno)d:%(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
    logging.info("train_test.py Start")
    SAMPLE = 50000
    p = PreProcess()

    train_filepath = 'data/train_1M.csv.out'
    #train_filepath = 'data/train_1000.csv.out'
    #test_filepath = 'data/test.csv.out'
    test_filepattern = 'data/test_%d_M.out'

    #Load train data
    logging.info("Loading train set...")
    X_train, y_train= p.load_train_data(train_filepath)
    
    #Sampling
    if y_train.shape[0] > SAMPLE:
        X_train = X_train[:SAMPLE]
        y_train = y_train[:SAMPLE]
    else:
        SAMPLE = y_train.shape[0]
    logging.info("Sampling %d" % SAMPLE)
    logging.info("Shape X_train = %r, y_train =%r" %(X_train.shape, y_train.shape ))
    logging.info("example X_train = %s\ny_train =%r" %(X_train[0], y_train[0]))
    logging.info("classes: %r" % list(np.unique(y_train)))
    
    #Training
    #Default
    C= 1.0
    gamma = 'auto'
    #For 1000 train
    #C= 1.0
    #gamma = 0.1
    
    #For 1m train
    #C = 0.10000000000000001
    #gamma = 0.0001 
    logging.info("Train SVC with C: %f, gamma: %f" %(C, gamma))
    svc = SVC(C=C, gamma = gamma, probability=True, verbose = True).fit(X_train, y_train)
    
    #Load test data for 5 times
    field = ['id', 'click']
    for part in range(1, 6):
        test_filepath = test_filepattern % part
        logging.info("Loading test set [%s]..." % test_filepath)
        X_test, ids_test= p.load_test_data(test_filepath)
        logging.info("Shape X = %r, ids =%r" %(X_test.shape, ids_test.shape ))
        logging.info("example X = %s\nids =%r" %(X_test[0], ids_test[0]))
        svc_probs = svc.predict_proba(X_test)
        #[prob of 0, prob of 1]
        logging.info("prob of test: %s" % svc_probs[:10])
        
        out_filepath = "%s-svc-t1M-s%d-c%f-g%f.csv" %(test_filepath, SAMPLE, C, gamma)
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

    logging.info("train_test.py End")
