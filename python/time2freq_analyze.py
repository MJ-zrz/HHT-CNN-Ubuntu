import matplotlib.pyplot as plt
from PyEMD import EMD, visualisation
import numpy as np
from scipy.signal import hilbert


''' time2freq_analyze.py
Functions:
1. emd_analyze;
2. emd_plot;
3. hhtlw
'''
''' The time sequence is characterized by the index of 'data'.
Therefore, 'data' -> 1-d array
'''
def emd_analyze(data):
    emd = EMD()
    IMFs = emd(data)
    return IMFs



def emd_plot(data, IMFs, mode="", title=""):
    plt.figure(figsize=(20, 15))
    plt.subplot(len(IMFs)+1, 1, 1)
    plt.plot(data, "r")
    plt.title("Original Signal")
    for num, imf in enumerate(IMFs):
        plt.subplot(len(IMFs)+1, 1, num+2)
        plt.plot(imf)
        plt.title(f"IMF {num+1}", fontsize=10)
    plt.subplots_adjust(hspace=0.8, wspace=0.2)
    if mode == 1:
        plt.show()
    elif isinstance(mode, str):
        if mode[-1] != "/":
            mode += "/"
        plt.savefig(f"{mode}{title}.png")



def hhtlw(IMFs,t,index,task_name,f_range=[0,500],t_range=[0,400],ft_size=[128,128],draw=1):
    fmin,fmax=f_range[0],f_range[1]         #时频图所展示的频率范围
    tmin,tmax=t_range[0],t_range[1]         #时间范围
    fdim,tdim=ft_size[0],ft_size[1]         #时频图的尺寸（分辨率）
    dt=(tmax-tmin)/(tdim-1)
    df=(fmax-fmin)/(fdim-1)
    vis = visualisation.Visualisation()
    #希尔伯特变化
    c_matrix=np.zeros((fdim,tdim))
    for imf in IMFs:
        imf=np.array([imf])
        #求瞬时频率
        freqs = abs(vis._calc_inst_freq(imf, t, order=False, alpha=None))
        #求瞬时幅值
        amp= abs(hilbert(imf))
        #去掉为1的维度
        freqs=np.squeeze(freqs)
        amp=np.squeeze(amp)
        #转换成矩阵
        temp_matrix=np.zeros((fdim,tdim))
        n_matrix=np.zeros((fdim,tdim))
        for i,j,k in zip(t,freqs,amp):
            if i>=tmin and i<=tmax and j>=fmin and j<=fmax:
                temp_matrix[round((j-fmin)/df)][round((i-tmin)/dt)]+=k
                n_matrix[round((j-fmin)/df)][round((i-tmin)/dt)]+=1
        n_matrix=n_matrix.reshape(-1)
        idx=np.where(n_matrix==0)[0]
        n_matrix[idx]=1
        n_matrix=n_matrix.reshape(fdim,tdim)
        temp_matrix=temp_matrix/n_matrix
        c_matrix+=temp_matrix
    
    t=np.linspace(tmin,tmax,tdim)
    f=np.linspace(fmin,fmax,fdim)

    plt.rcParams["figure.figsize"] = (6.4, 4.8)
    #可视化
    if draw==1:
        fig,axes=plt.subplots()
        plt.rcParams['font.sans-serif']='Times New Roman'
        plt.contourf(t, f, c_matrix,cmap="jet")
        plt.xlabel('Time/s',fontsize=16)
        plt.ylabel('Frequency/Hz',fontsize=16)
        plt.title('Hilbert spectrum',fontsize=20)
        x_labels=axes.get_xticklabels()
        [label.set_fontname('Times New Roman') for label in x_labels]
        y_labels=axes.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in y_labels]
        # plt.show()
        plt.savefig(f"../fig/{task_name}/ht/ht-{index}.png")
    return t,f,c_matrix


