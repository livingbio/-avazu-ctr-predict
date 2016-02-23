import numpy as np
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

    def load_train_data(self, filepath):
        with open(filepath) as ifile:
            #MAKE numpy array
            reader = csv.reader(ifile)
            x = list(reader)
            logging.debug('small_x %s' %x)
            X = np.array(x).astype('int64')
            #X = np.array(x)
            #Get click
            y = X[:,1]
            #Remove id and click
            X = X[:,2:]
            return X, y

    #TODO Add load_test_data

if __name__ == "__main__":
    p = PreProcess()
    filepath = 'data/train_10.csv'
    out_filepath = p.convert(filepath)
    X, y = p.load_train_data(out_filepath)
    logging.info("Shape X = %r, y =%r" %(X.shape, y.shape ))
    logging.info("example X = %s\ny =%r" %(X[0], y[0]))

