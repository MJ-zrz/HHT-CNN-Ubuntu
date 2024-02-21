from data_preprocess_MIT_BIH import *
from data_preprocess_MIMIC_III import *

def data_fetch(task_name):
    # For different datasets, particular preprocessing should be written respectively.
    if task_name == "MIMIC-III":
        data_num, time_length, y, X = pulsedata_preprocess(task_name)
    elif task_name == "MIT-BIH":
        data_num, time_length, y, X = loadData(task_name)
    return data_num, time_length, y, X
