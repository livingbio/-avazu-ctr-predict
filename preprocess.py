import numpy as np
from sklearn.preprocessing import OneHotEncoder
import logging
import csv

logging.basicConfig(format='--%(asctime)s:[%(levelname)s]:%(lineno)d:%(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)

class PreProcess:
    def convert(self, filepath):
        with open(filepath) as ifile, open(filepath+'.out', 'w') as ofile:
            #First row is column names
            hex_list = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model']
            reader = csv.DictReader(ifile)
            field_name = reader.fieldnames
            logging.info('read fieldnames %s' % field_name)

            new_field = ['dayOfWeek', 'day', 'date']
            field_name += new_field
            writer = csv.DictWriter(ofile, field_name)
            #Don't write header
            #writer.writeheader()
            logging.info('write fieldnames %s' % field_name)

            for row in reader:
                #TODO add row_count, and print
                hour = row['hour']
                #logging.debug('hour %s' % hour)
                if len(hour) !=8:
                    logging.warning('Wrong format: hour %s' % hour)
                    continue
                row ['date'] = int(hour[2:6])
                row ['day'] = int(hour[2:4])
                row ['dayOfWeek'] = row['day'] % 7
                row ['hour'] = int(hour[4:6])
                #Make hex2int
                for key in hex_list:
                    row[key] = int(row[key], 16)
                #Remove id for trainining
                row['id'] = 0
                writer.writerow(row)
            out_filepath = filepath + '.out'
            logging.info("Outfile path %s" % out_filepath)
            return out_filepath

    def load_train_data(self, filepath, regression = False, category= False):
        with open(filepath) as ifile:
            #MAKE numpy array
            reader = csv.reader(ifile)
            x = list(reader)
            logging.debug('small_x %s' %x)
            X = np.array(x)
            #Get click
            if regression:
                y = X[:,1].astype('float')
            else:
                y = X[:,1].astype('int8')
            #Remove id and click
            X = X[:,2:].astype('int64')
            #FIXME memory error, map to smaller index

            if category:
                #C20 had -1, make it 1
                for i in X:
                    if i[20] == -1:
                        i[20] = 0
                logging.info("after X = %s" %(X))
                
                #the index of category features
                #25 features, only C15, C16 is value
                cat_index = range(0, 15) + range(17, 25)
                enc = OneHotEncoder(categorical_features=cat_index, dtype=np.int64, handle_unknown='ignore')
                enc.fit(X)
                X = enc.transform(X)
                return X, y, enc
            else:
                return X, y

    def load_test_data(self, filepath, enc = None):
        with open(filepath) as ifile:
            #MAKE numpy array
            reader = csv.reader(ifile)
            x = list(reader)
            logging.debug('small_x %s' %x)
            X = np.array(x)
            #X = np.array(x)
            #Get id
            ids = X[:,0]
            X = X[:,1:].astype('int64')
            if enc !=None:
                #C20 had -1, make it 1
                for i in X:
                    if i[20] == -1:
                        i[20] = 0
                logging.info("after X = %s" %(X))
                
                X=enc.transform(X)

            return X, ids

    def divide_train_data(self, filepath):
        #Divide training data label 1 vs 0
        with open(filepath) as ifile, open(filepath+'.1', 'wb') as ofile_1, open(filepath+'.0', 'wb') as ofile_0:
            #MAKE numpy array
            reader = csv.reader(ifile)
            writer_0 = csv.writer(ofile_0)
            writer_1 = csv.writer(ofile_1)
            cnt_0 = cnt_1 = 0
            for row in reader:
                #row is list, row[1] is str
                #[1] is click
                if row[1] == '1':
                    writer_1.writerow(row)
                    cnt_1 +=1
                elif row[1] == '0':
                    writer_0.writerow(row)
                    cnt_0 +=1

            logging.info('count of \'0\' : %d in file %s' %(cnt_0, filepath))
            logging.info('count of \'1\' : %d' %cnt_1)

if __name__ == "__main__":
    p = PreProcess()

    #filepath = 'data/train_10.csv'
    #out_filepath = p.convert(filepath)

    out_filepath = 'data/train_10.csv.out'
    X, y, enc = p.load_train_data(out_filepath, regression=True, category = True)
    logging.info("Shape X = %r, y =%r" %(X.shape, y.shape ))
    logging.info("example X = %s\ny =%s" %(X[0], y[0]))
    '''
    test_filepath = 'data/test_10.csv.out'
    X, ids = p.load_test_data(test_filepath, enc = enc)
    logging.info("Shape X = %r, ids =%r" %(X.shape, ids.shape ))
    logging.info("example X = %s\nids =%r" %(X[0], ids[0]))
    '''
    #train_filepath = 'data/train_s404_100K.out'
    #p.divide_train_data(train_filepath)
