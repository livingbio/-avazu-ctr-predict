import csv
import json
from datetime import datetime

def convert(filepath):
	with open(filepath) as ifile, open(filepath+'.out', 'w') as ofile:	
		reader = csv.DictReader(ifile)
		writer = csv.DictWriter(ofile, reader.fieldnames)

		for row in reader:
			hour = row['hour']
			year = int('20'+hour[:2])
			month = int(hour[2:4])
			day = int(hour[4:6])
			hour = int(hour[6:8])

			row['hour'] = datetime(year, month, day, hour, 0).strftime("%Y-%m-%d %H:%M:%S")
			writer.writerow(row)

if __name__ == "__main__":
	import clime.now
