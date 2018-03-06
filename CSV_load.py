import csv
import numpy as np
from config import args


def csv_loader(file_names):
    y_main = []
    for img_name in file_names:
        y = []
        with open(args.gaze_path + img_name, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            reader.next()
            for row in reader:
                y.append(int(row[0]))
                y.append(int(row[1]))
        y_main.append(y)
    rows = max([len(l) for l in y_main])
    columns = len(y_main)
    y_ = np.zeros(shape=[columns, rows], dtype=int)
    for row_idx, row in enumerate(y_main):
        y_[row_idx, :len(row)] = np.array(row)
    return np.transpose(y_)


if __name__=='__main__':
    csv_loader()
