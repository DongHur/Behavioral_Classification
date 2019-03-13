import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from Code.utils.kmeans_pytorch.kmeans import lloyd, lloyd_batch, kmeans_core

def new_func():
    data_test = np.load('../../Data/PowerSpectogram_Data/CB_power_ALLdata_60bp_25fBin.npy')
    bp,coord,fbin,num_data = data_test.shape
    data_cluster = data_test.reshape((bp*coord*fbin, data_test.shape[-1])).transpose(1,0)
    # YOU MUST RANDOMIZE YOUR DATA BEFORE YOU SEND YOUR DATA INTO BATCH!!!
    min_k = 2
    max_k = 20
    sse_25 = np.zeros(max_k-min_k+1)
    for k_i, k in enumerate(range(min_k, max_k+1)):
        km = kmeans_core(k=k, data_array=data_cluster, device=2, batch_size=10000, epochs=10, all_cuda=False, random_state=2)
        sse_25[k_i] = km.run()

    k_list_25 = range(min_k, max_k+1)
    plt.plot(k_list,sse_25)
    plt.show()

def main():
    bp,coord,fbin,num_data = data_test.shape
    data_test = np.load('../../Data/PowerSpectogram_Data/CB_power_100data_60bp_50fBin.npy')
    data_cluster = data_test.reshape((bp*coord*fbin, data_test.shape[-1])).transpose(1,0)
    # YOU MUST RANDOMIZE YOUR DATA BEFORE YOU SEND YOUR DATA INTO BATCH!!!
    min_k = 2
    max_k = 20
    sse = np.zeros(max_k-min_k+1)
    for k_i, k in enumerate(range(min_k, max_k+1)):
        km = kmeans_core(k=k, data_array=data_cluster, device=2, batch_size=10000, epochs=10, all_cuda=False, random_state=2)
        sse[k_i] = km.run()
    k_list = range(min_k, max_k+1)
    plt.plot(k_list,sse)
    plt.show()

if __name__ == "__main__":
    main()