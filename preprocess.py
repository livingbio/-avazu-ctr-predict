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
            #TODO dayOfWeek
            #new_field = ['dayOfWeek', 'day', 'date']
            new_field = ['day', 'date']
            field_name += new_field
            writer = csv.DictWriter(ofile, field_name)
            logging.info('fieldnames %s' % field_name)
            for row in reader:
                hour = row['hour']
                row ['date'] = int(hour[2:6])
	        row ['day'] = int(hour[2:4])
	        row ['hour'] = int(hour[4:6])
                for key in hex_list:
                    continue

                writer.writerow(row)

        return filepath+'.out'

if __name__ == "__main__":
    p = PreProcess()
    filepath = 'data/train_10.csv'
    logging.info(p.convert(filepath))

