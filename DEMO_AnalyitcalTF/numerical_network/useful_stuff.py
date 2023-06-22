import csv
import numpy as np
def read_csv_file(file_path):
    file = open(file_path)
    csvreader = csv.reader(file)
    rows = []

    for row in csvreader:
        rows.append(row)

    file.close()

    rows=np.array(rows)
    mossy_input = rows[:,0]
    f_mean = rows[:,1]
    #f_sd = rows[:,2]

    #mossy_input[i] = type string --> need to map into an array of type float
    mossy_input = list(map(float, mossy_input))
    f_mean = list(map(float, f_mean))
    #f_sd = list(map(float, f_sd))

    return np.array(mossy_input), np.array(f_mean) #,np.array(f_sd)
