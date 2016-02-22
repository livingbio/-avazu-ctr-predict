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
            #TODO dayOfWeek
            new_field = ['dayOfWeek', 'day', 'date']
            #new_field = ['day', 'date']
            field_name += new_field
            writer = csv.DictWriter(ofile, field_name)
            writer.writeheader()
            logging.info('write fieldnames %s' % field_name)
            #TODO MAKE numpy array
            #reader_2 = csv.reader(ifile)
            #x = list(reader_2)
            #X = np.array(x)
            #.astype('int64')
            #y = X[:,1]

            for row in reader:
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
                writer.writerow(row)

            logging.info(filepath)
            #return X, y

if __name__ == "__main__":
    p = PreProcess()
    filepath = 'data/train_10.csv'
    logging.info(p.convert(filepath))

