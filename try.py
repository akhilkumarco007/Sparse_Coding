# from CSV_load import *
# import os
# import numpy as np
#
# file_names = os.listdir(args.gaze_path)
# y_ = csv_loader(file_names)
#
# y = np.transpose(y_)
# y_1 = y.astype(float)
#
# for i in range(len(y_1)):
#     max = float(np.max(y_1[i]))
#     min = float(np.min(y_1[i]))
#     y_1[i] = (y_1[i] - min) / float(max - min)
#
# y_2 = np.round(y_1, 3)
#
# maxi = (np.where(y != 0))
# mini = (np.where(y != 0))
# print(maxi, mini)
#
# print()

import numpy as np
import csv
from sklearn.cluster import AgglomerativeClustering

x = np.array([[1, 2, 3, 5, 6, 0, 0, 0], [10, 11, 12, 0, 0, 0, 0, 0]])

print(x * 3 + 0)