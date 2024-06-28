import os
import pandas as pd
import numpy as np

# read_directory = "data/angrymen/quad-AngryMen/gold_labels/original/"
# write_directory = "data/angrymen/quad-AngryMen/gold_labels/final/"

read_directory = "data/debatepedia/qem_debatepedia/gold_labels/original/"
write_directory = "data/debatepedia/qem_debatepedia/gold_labels/final/"

pd.options.display.float_format = "{:,.4f}".format
for filename in os.listdir(read_directory):
    data = pd.read_csv(read_directory + filename, header=None)
    data[1] = data[1].apply(lambda x:round(x,4))
    data = data.drop(data.columns[0], axis=1)
    filename_split= filename.split("_")
    filename= filename_split[-1]
    filename = filename.replace('.csv', '.txt')
    data.to_csv(write_directory + filename, header=None, index=None)
