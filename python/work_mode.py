import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

from data_fetch import *
from LeNet import *
from ResNet import *



def draw_mode(task_name, pulse_freq_max, crop_size_list, resize_size_list):
    
    data_num, time_length, Y, X = data_fetch(task_name)
    t = np.array([i for i in range(time_length)])
    X = np.reshape(X, (-1, time_length))

    if not os.path.exists("../fig/"):
        os.mkdir("../fig/")
    if not os.path.exists(f"../fig/{task_name}/"):
        os.mkdir(f"../fig/{task_name}/")
    if not os.path.exists(f"../fig/{task_name}/imf/"):
        os.mkdir(f"../fig/{task_name}/imf/")
    if not os.path.exists(f"../fig/{task_name}/ht/"):
        os.mkdir(f"../fig/{task_name}/ht/")
    if not os.path.exists(f"../fig/{task_name}/ht-resize-{resize_size_list[0]}x{resize_size_list[1]}/"):
        os.mkdir(f"../fig/{task_name}/ht-resize-{resize_size_list[0]}x{resize_size_list[1]}/")
    


    ''' 3. Time-freq analysis.
    '''
    num = 0
    while num < data_num:  # Relieve memory pressure.
        limit = min(data_num, num + 1000)
        for i in range(num, limit):
            data = X[i, :]
            IMFs = emd_analyze(data)
            # emd_plot(data, IMFs, 1)
            emd_plot(data, IMFs, f"../fig/{task_name}/imf/", f"data-{i}")
            _ = hhtlw(IMFs, t, i, task_name, f_range=[0, pulse_freq_max], t_range=[0, time_length], ft_size=[128, time_length], draw=1)
            input_fig_path = f"../fig/{task_name}/ht/ht-{i}.png"
            output_fig_path = f"../fig/{task_name}/ht-resize-{resize_size_list[0]}x{resize_size_list[1]}/ht-resize-{i}.png"
            image_crop(input_fig_path, output_fig_path, crop_size_list)
            image_resize(output_fig_path, output_fig_path, resize_size_list)
        num += 1000

    with open(f"../fig/{task_name}/Y.txt", "w", encoding="utf-8") as f:
        f.write("\n".join( map(str, Y) ))
    
    
    
def train_mode(task_name, resize_size_list, model_name="lenet", batch_size=32, train_iter_num=100, test_size=0.2, lr=0.001, optimizer="adamw"):

    X_data_path = f"../fig/{task_name}/ht-resize-{resize_size_list[0]}x{resize_size_list[1]}/"
    data_num = len( os.listdir(X_data_path) )
    X_data_shape = (data_num, 3, *resize_size_list)
    X_dataset = np.zeros(X_data_shape, dtype=float)
    for i in range(data_num):
        image = Image.open(f"{X_data_path}ht-resize-{i}.png")
        data = np.array(image)
        X_dataset[i, :, :, :] = data.reshape(3, *resize_size_list)

    Y = []
    with open(f"../fig/{task_name}/Y.txt", "r", encoding="utf-8") as f:
        Y = f.readlines()
    label_num = len(set(Y))
    for i in range(len(Y)):
        Y[i] = int(float(Y[i].strip()))
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_dataset, Y, test_size=test_size)
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)

    train_dataset = PreLoader(X_train, Y_train)
    test_dataset = PreLoader(X_test, Y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if model_name.lower() == "lenet":
        model = LeNet()
        model_path = f"../model/{task_name}/lenet.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name.lower() == "resnet18":
        model = resnet18(label_num)
        model_path = f"../model/{task_name}/resnet18.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name.lower() == "resnet34":
        model = resnet34(label_num)
        model_path = f"../model/{task_name}/resnet34.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name.lower() == "resnet50":
        model = resnet50(label_num)
        model_path = f"../model/{task_name}/resnet50.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name.lower() == "resnet101":
        model = resnet101(label_num)
        model_path = f"../model/{task_name}/resnet101.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    batch_loss_list = []
    if optimizer.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    train_batch_num = math.ceil( (1-test_size) * data_num / batch_size )
    for i in range(train_iter_num):
        loss_temp = 0
        for j, (batch_data,batch_label) in enumerate(train_loader):
            if torch.cuda.is_available():
                batch_data,batch_label = batch_data.cuda(),batch_label.cuda()
            optimizer.zero_grad()
            prediction = model(batch_data)
            loss = loss_func(prediction,batch_label)
            batch_loss_list.append(loss)
            loss_temp += loss
            loss.backward()
            optimizer.step()
            if (j + 1) % train_batch_num == 0:
                print('Epoch-%d-batch-%d-loss: %.9f' % (i+1, j+1, loss_temp/batch_size))
                loss_temp = 0
    
    end_time = time.time()
    print('Total training time: %d s' % int((end_time-start_time)))
    if torch.cuda.is_available():
        batch_loss_list = torch.Tensor(batch_loss_list).cpu()
    else:
        batch_loss_list = torch.Tensor(batch_loss_list).detach().numpy()
    plt.figure(constrained_layout=True)
    plt.subplot(2,1,1)
    plt.plot(batch_loss_list[1:])
    plt.title("Loss curve")
    plt.xlabel("Epoch*Batch")
    plt.ylabel("Loss")
    epoch_loss_list = batch_loss_list[::batch_size]
    plt.subplot(2,1,2)
    plt.plot(epoch_loss_list[1:])
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.show()
    plt.savefig(f"../fig/{task_name}/loss_curve.png")
    
    correct = 0
    for batch_data,batch_label in train_loader:
        if torch.cuda.is_available():
            batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
        prediction = model(batch_data)
        predicted = torch.max(prediction.data, 1)[1]
        correct += (predicted == batch_label).sum()
    print(f"Training accuracy: {(correct / ((1-test_size)*data_num))}")
    correct = 0
    for batch_data,batch_label in test_loader:
        if torch.cuda.is_available():
            batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
        prediction = model(batch_data)
        predicted = torch.max(prediction.data, 1)[1]
        correct += (predicted == batch_label).sum()
    print(f"Testing accuracy: {(correct / (test_size*data_num))}")

    if not os.path.exists(f"../model/"):
        os.mkdir(f"../model/")
    if not os.path.exists(f"../model/{task_name}/"):
        os.mkdir(f"../model/{task_name}/")

    torch.save(obj=model.state_dict(), f=f"../model/{task_name}/{model_name.lower()}.pth")
    
    
    
def test_mode(task_name, resize_size_list, model_name="lenet", batch_size=32):
    
    X_data_path = f"../fig/{task_name}/ht-resize-{resize_size_list[0]}x{resize_size_list[1]}/"
    data_num = len( os.listdir(X_data_path) )
    X_data_shape = (data_num, 3, *resize_size_list)
    X_dataset = np.zeros(X_data_shape, dtype=float)
    for i in range(data_num):
        image = Image.open(f"{X_data_path}ht-resize-{i}.png")
        data = np.array(image)
        X_dataset[i, :, :, :] = data.reshape(3, *resize_size_list)
    X_dataset = torch.Tensor(X_dataset)

    Y = []
    with open(f"../fig/{task_name}/Y.txt", "r", encoding="utf-8") as f:
        Y = f.readlines()
    label_num = len(set(Y))
    for i in range(len(Y)):
        Y[i] = int(float(Y[i].strip()))    

    if model_name.lower() == "lenet":
        model = LeNet()
        model_path = f"../model/{task_name}/lenet.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    elif model_name.lower() == "resnet18":
        model = resnet18(label_num)
        model_path = f"../model/{task_name}/resnet18.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    elif model_name.lower() == "resnet34":
        model = resnet34(label_num)
        model_path = f"../model/{task_name}/resnet34.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    elif model_name.lower() == "resnet50":
        model = resnet50(label_num)
        model_path = f"../model/{task_name}/resnet50.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    elif model_name.lower() == "resnet101":
        model = resnet101(label_num)
        model_path = f"../model/{task_name}/resnet101.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    test_dataset = PreLoader(X_dataset, Y)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    for batch_data,batch_label in test_loader:
        prediction = model(batch_data)
        predicted = torch.max(prediction.data, 1)[1]
        correct += (predicted == batch_label).sum()
    print(f"Testing accuracy: {(correct / data_num)}")


