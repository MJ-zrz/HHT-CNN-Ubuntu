import json

from work_mode import *

''' main.py
Steps:
1. Get the task config info;
2. Drawing / Training / Testing / Drawing + Testing.
'''



if __name__ == "__main__":

    ''' 1. Get the task config info.
    '''
    config_dict = {}
    with open("./config.json") as f:
        config_dict = json.load(f)
    task_name = config_dict["Task name"]
    time_freq_method = config_dict["Time-freq method"]
    pulse_freq_max = config_dict["Max pulse (Hz)"]
    crop_size_list = config_dict["Cropping size"]
    resize_size_list = config_dict["Image size"]
    batch_size = config_dict["Batch size"]
    model_name = config_dict["Model"]
    mode = config_dict["Mode"]
    train_iter_num = config_dict["Train iter num"]
    test_size = config_dict["Test size"]
    lr = config_dict["Learning rate"]
    optimizer = config_dict["Optimizer"]

    ''' 2. Drawing / Training / Testing / Drawing + Testing.
    '''
    if mode == "draw":
        print("===================== Drawing mode ======================")
        draw_mode(task_name, pulse_freq_max, crop_size_list, resize_size_list)
    elif mode == "train":
        print("===================== Training mode =====================")
        train_mode(task_name, resize_size_list, model_name, batch_size, train_iter_num, test_size, lr, optimizer)
    elif mode == "test":
        print("===================== Testing mode ======================")
        test_mode(task_name, resize_size_list, model_name, batch_size)
    elif mode == "draw-train":
        print("===================== Drawing mode ======================")
        draw_mode(task_name, pulse_freq_max, crop_size_list, resize_size_list)
        print("===================== Training mode =====================")
        train_mode(task_name, resize_size_list, model_name, batch_size, train_iter_num, test_size, lr, optimizer)
    else:
        print("===================== Drawing mode ======================")
        draw_mode(task_name, pulse_freq_max, crop_size_list, resize_size_list)
        print("===================== Training mode =====================")
        train_mode(task_name, resize_size_list, model_name, batch_size, train_iter_num, test_size, lr, optimizer)
        

