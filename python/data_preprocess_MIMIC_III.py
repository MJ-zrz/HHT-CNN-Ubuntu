import numpy as np
import pandas as pd


''' ver1
'''
# def pulsedata_preprocess(filepath):
#     df = pd.read_csv(filepath)
#     data_dict = dict()
#     index_list = df["index"].tolist()
#     label_list = df["label"].tolist()
#     data_dict["index"] = index_list
#     data_dict["label"] = label_list
#     data_dict["time_length"] = len(df.leys())-2
#     for i in range(1, len(df.keys())-1):
#         data_dict[i] = df[f"{i}"].tolist()
#     return data_dict


''' ver2
'''
# def pulsedata_preprocess(filepath):
#     df = pd.read_csv(filepath)
#     data_dict = dict()
#     index_list = df["index"].tolist()
#     label_list = df["label"].tolist()
#     data_dict["index"] = index_list
#     data_dict["label"] = label_list
#     data_dict["time_length"] = len(df.keys())-2
#     for i in range(1, len(df.keys())-1):
#         data_dict[i] = df[f"{i}"].tolist()
#     return data_dict


# def pulsedata_dimension_change(data_dict):
#     data_dict_ret = dict()
#     time_length = data_dict["time_length"]
#     data_dict_ret["time_length"] = time_length
#     data_dict_ret["label"] = data_dict["label"]
#     for i in range( len(data_dict["index"]) ):
#         data_dict_ret[i] = [data_dict[j+1][i] for j in range(time_length)]
#     return data_dict_ret


''' ver3
'''
def pulsedata_preprocess(task_name):
    filepath = f"../data/{task_name}/PulseData.csv"
    df = pd.read_csv(filepath)
    index_list = df["index"].tolist()
    data_num = len(index_list)
    time_length = len(df.keys())-2
    labels = np.array( df["label"] )
    data_array = np.zeros((data_num, time_length))
    for i in range(time_length):
        data_array[:, i] = np.array(df[f"{i+1}"])
    return data_num, time_length, labels, data_array



if __name__ == "__main__":
    filepath = "../data/PulseData.csv"
    data_num, time_length, labels, data_array = pulsedata_preprocess(filepath)
    print(data_num)
    print(time_length)
    print(labels)
    print(data_array[10, :])


