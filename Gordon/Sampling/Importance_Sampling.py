import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage import morphology
from sklearn.neighbors import NearestNeighbors



class Sampling():
    def __init__(self, X, pwr, k_nearest, num_points):
        # X - (N x 2) || num_points - *odd* dimension of watershed
        # pwr - (N x D)
        self.X = X
        self.pwr = torch.Tensor(pwr)
        self.N = X.shape[0]
        self.D = pwr.shape[1]
        self.k_nearest = k_nearest
        self.num_points = self._confirm_num_points(num_points)
        self.maxL = (X.reshape(-1).abs().max().ceil()+1).item()
        self.xx = None
        self.yy = None
    def _confirm_num_points(self, num_points):
        if num_points%2 ==0:
            return num_points+1
        else:
            return num_points
    def _fft_convolve2d(self,x,y):
        fr = np.fft.fft2(x)
        fr2 = np.fft.fft2(y)
        cc = np.fft.fftshift(np.real(np.fft.ifft2(fr*fr2)))
        return cc
    def _label_points_to_watershed(self, watershed_dist):
        X_idx = torch.argmin((self.X[:,0]-self.xx.unsqueeze(-1).expand(self.num_points,self.N)).abs(), dim=0)
        Y_idx = torch.argmin((self.X[:,1]-self.yy.unsqueeze(-1).expand(self.num_points,self.N)).abs(), dim=0)
        labels = watershed_dist[X_idx, Y_idx]
        return labels
    
    def knn(self):
        # X - (N x D) || K_idx - (N x K)  || K_matrix_idx - (N x N)  || K_dist - (N x K)
        nbrs = NearestNeighbors(n_neighbors=self.k_nearest+1, algorithm='kd_tree').fit(self.X)
        K_dist, K_idx = nbrs.kneighbors(self.X)
        K_matrix_idx = nbrs.kneighbors_graph(self.X).toarray()
        return torch.Tensor(K_idx), torch.Tensor(K_matrix_idx), torch.Tensor(K_dist)
    
    def gaussian_conv(self, K_dist):
        sigma = K_dist[:,-1].median()
        print("sigma: ", sigma.item())
        L_bound = -1.0*self.maxL
        U_bound = 1.0*self.maxL
        xx = torch.linspace(L_bound, U_bound, self.num_points)
        self.xx = xx
        yy = torch.linspace(L_bound, U_bound, self.num_points)
        self.yy = yy
        XX, YY = torch.meshgrid((xx, yy))
        # GAUSSIAN KERNEL
        G = torch.exp(-0.5*(XX**2 + YY**2)/sigma**2)/(2*np.pi*sigma**2);
        plt.imshow(G, extent=[L_bound, U_bound, L_bound, U_bound])
        plt.title("Gaussian Kernel")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()
        # DATA HISTOGRAM
        H, xedges, yedges = np.histogram2d(self.X[:,0].numpy(), self.X[:,1].numpy(), self.num_points, [[L_bound,U_bound],[L_bound,U_bound]])
        H = torch.Tensor(H/H.sum())
        plt.imshow(H, extent=[L_bound, U_bound, L_bound, U_bound]) 
        plt.title("Data Histogram")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()
        
        # CONVOLUTION USING FFT (NUMPY)
        GH_conv = self._fft_convolve2d(G,H)
        GH_conv[GH_conv<0] = 0
    
        plt.imshow(GH_conv, extent=[L_bound, U_bound, L_bound, U_bound])
        plt.title("Gaussian Kernel Convolution w/ Data Histogram")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show() 
        return GH_conv
    
    def apply_watershed(self, GH_conv, min_distance=6):
        # ALGORITHM
        thresh = threshold_otsu(GH_conv)

        binarize_GH_conv = 1.0*(GH_conv>thresh)
        plt.imshow(binarize_GH_conv)
        plt.title("Binarized Gaussian Conv. on Data")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()

        ndi = ndimage.distance_transform_edt(binarize_GH_conv)
        plt.imshow(ndi)
        plt.title("Distance Transformation on Binarized Graph")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()

        local_max = peak_local_max(ndi, indices=False, min_distance=min_distance)
        plt.imshow(1.0*local_max)
        plt.title("Peak Local Max")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()

        markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
        plt.imshow(1.0*markers)
        plt.title("Watershed Markers")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()

        invert_ndi = -1.0*ndi
        watershed_dist = morphology.watershed(invert_ndi, markers,
                                            connectivity=ndimage.generate_binary_structure(2, 2),
                                            watershed_line=True)
        plt.imshow(watershed_dist)
        plt.title("Watershed")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()
        # FIND GROUP LABEL FOR EACH POINTS
        labels = self._label_points_to_watershed(watershed_dist)
        return watershed_dist, labels
    def create_template(self, labels):
        Template={}
        for i in np.arange(1,max(labels)+1):
            Template[i] = self.pwr[np.where(labels==i),:].squeeze(dim=0)
        return Template
    def importance_sampling(self, Template, frac_training_set=0.3):
        # frac_training_set SHOULD NOT BE 1; SHOULD BE A FRACTION LESS THAN 1
        # GET PARAMETERS FOR PROPORTIONALLY SAMPLING FROM EACH WATERSHED
        len_group = torch.Tensor([Template[i].shape[0] for i in range(1,len(Template)+1)])
        num_in_group = torch.round(frac_training_set*self.N*len_group/len_group.sum())
        # PROPER INDEXING OF NEW SAMPLES BASED ON PROPORTION SAMPLED FROM EACH WATERSHED
        num_training = int(num_in_group.round().sum().item())
        training_data = torch.zeros(num_training, self.D)
        group_cum_sum = torch.cumsum(num_in_group, dim=0)
        group_idx = torch.cat((torch.Tensor([0]),group_cum_sum))
        # RANDOMLY SAMPLE FROM EACH WATERSHED BASED ON PROPORTION
        for i in range(len_group.shape[0]):
            idx = torch.randperm(int(len_group[i].item()))[0:int(num_in_group[i].item())]
            training_data[int(group_idx[i]):int(group_idx[i+1]), :] = Template[i+1][idx,:]
        return training_data