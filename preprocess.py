import numpy as np
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
import logging
import csv

#logging.basicConfig(format='--%(asctime)s:[%(levelname)s]:%(lineno)d:%(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)

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

    def transform_and_map(self, X, ignore_index = None):
        features = np.shape(X)[1]
        #Init
        map_dict = [None] * features
        trans_X = X.copy()
        for i in range(features):
            if i not in ignore_index:
                map_dict[i], trans_X[:,i] = np.unique(X[:,i], return_inverse=True)
                logging.info("Dict %d(%d) : %s" %(i, len(map_dict[i]), map_dict[i]))

        return trans_X, map_dict

    def transform_with_map(self, X, map_dict, ignore_index = None):
        features = np.shape(X)[1]
        #Init
        trans_X = X.copy()
        for i in range(features):
            if i not in ignore_index:
                #TODO 
                #replace value with index in map
                #trans_X[:,i] = X[:,i]
                continue

        return trans_X, map_dict

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
            if category:
                #Remove id and click
                X = X[:,2:].astype('int64')
                logging.info("X = \n%s" %X[:3])
                #25 features, only C15, C16 is value
                ignore_index = [15, 16]
                cat_index = range(0, 15) + range(17, 25)
                """
                X, map_dict = self.transform_and_map(X, ignore_index = ignore_index)
                logging.info("After small index transform X = \n%s" %X[:3])
                enc = OneHotEncoder(categorical_features=cat_index, dtype=np.int8, handle_unknown='ignore',n_values='auto')
                X = enc.transform(X)
                """
                map_dict = range(25)
                enc = LabelBinarizer()
                logging.info("Shape X = %r, %r" %(X.shape))
                new_X = X[:, ignore_index]
                for i in cat_index:
                    logging.info("Shape X[:,%d] = %r" %( i, X[:,i].shape))
                    logging.debug("X[:,%d] = %s" % (i, X[:10,i]))
                    enc.fit(X[:,i])
                    #logging.info("enc.get_params = %s" %enc.get_params())
                    new_X = np.concatenate((new_X, enc.transform(X[:,i])), axis=1)
                    logging.debug("After transform X[:,%d] = %s" % (i, enc.transform(X[:10,i])))
                    map_dict[i] = enc.classes_
                    #FIXME len(dict) =2 will have problem
                    logging.debug("After transform shape X[:,%d] = %r" % (i, enc.transform(X[:10,i]).shape))
                    logging.info("Dict %d(%d) : %s" %(i, len(map_dict[i]), map_dict[i]))
                
                logging.info("Shape new X = %r, %r" %(new_X.shape))
                logging.info("After enc transform X[0] =\n%s" %new_X[0])
                return new_X, y, enc, map_dict
            else:
                #Remove id and click
                X = X[:,2:].astype('int64')
                return X, y

    def load_test_data(self, filepath, enc = None, map_dict = None):
        with open(filepath) as ifile:
            #MAKE numpy array
            reader = csv.reader(ifile)
            x = list(reader)
            logging.debug('small_x %s' %x)
            X = np.array(x)
            #Get id
            ids = X[:,0]
            X = X[:,1:].astype('int64')
            if enc !=None and map_dict !=None:
                logging.info("X = \n%s" %X[:3])
                #25 features, only C15, C16 is value
                cat_index = range(0, 15) + range(17, 25)
                ignore_index = [15, 16]
                """
                X = self.transform_with_map(X, map_dict, ignore_index = ignore_index)
                logging.info("After small index transform X = \n%s" %X[:3])
                X = enc.transform(X)
                logging.info("After enc transform X[0] =\n%s" %X.getrow(0))
                """

                logging.info("Shape X = %r, %r" %(X.shape))
                new_X = X[:, ignore_index]
                for i in cat_index:
                    logging.info("Shape X[:,%d] = %r" %( i, X[:,i].shape))
                    logging.debug("X[:,%d] = %s" % (i, X[:10,i]))
                    enc.fit(map_dict[i])
                    new_X = np.concatenate((new_X, enc.transform(X[:,i])), axis=1)
                    logging.debug("After transform X[:,%d] = %s" % (i, enc.transform(X[:10,i])))
                    logging.debug("After transform shape X[:,%d] = %r" % (i, enc.transform(X[:10,i]).shape))
                    #FIXME len(dict) =2 will have problem
                    logging.info("Dict %d(%d) : %s" %(i, len(map_dict[i]), map_dict[i]))
                
                logging.info("Shape new X = %r, %r" %(new_X.shape))
                logging.info("After enc transform X[0] =\n%s" %new_X[0])
                return new_X, ids
            
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

    logging.basicConfig(format='--%(asctime)s:[%(levelname)s]:%(lineno)d:%(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.DEBUG)
    #filepath = 'data/train_10.csv'
    #out_filepath = p.convert(filepath)

    #out_filepath = 'data/train_10.csv.out'
    out_filepath = 'data/train_1000.csv.out'
    X, y, enc, map_dict = p.load_train_data(out_filepath, regression=True, category = True)
    logging.info("Shape X = \n%r, y =%r" %(X.shape, y.shape ))
    
    test_filepath = 'data/test_1000.csv.out'
    X, ids = p.load_test_data(test_filepath, enc = enc, map_dict = map_dict)
    logging.info("Shape X = \n%r, ids =%r" %(X.shape, ids.shape ))
    #logging.info("example X = \n%s\nids =%r" %(X[0], ids[0]))
    
    #train_filepath = 'data/train_s404_100K.out'
    #p.divide_train_data(train_filepath)
