import torch
from torch import Tensor
import numpy as np
from utils.kmeans_pytorch.pairwise import pairwise_distance
# import math
from tqdm import trange

from scipy import linalg, sparse
from sklearn.utils import check_random_state
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances

CUDA = torch.cuda.is_available()

class spectral_core:
    def __init__(self, k, data_array, sigma=0.3 device=0, batch_size=1000, epochs=200, all_cuda=True, random_state=0, decrease_k=False):
        self.k = k
        self.data_array = Laplacian(data_array, K=k, sigma=sigma)
        self.data = Tensor(self.data_array,)
        self.device = device
        self.all_cuda = all_cuda
        if all_cuda and CUDA:
            self.data = self.data.cuda(device)
        
        self.dim = data_array.shape[-1]
        self.data_len = data_array.shape[0]
        self.cent = Tensor(_k_init(X=data_array, n_clusters=k, x_squared_norms=row_norms(data_array, squared=True), random_state=check_random_state(random_state)))

        if CUDA:
            self.cent = self.cent.cuda(device)
            
        self.epochs = epochs
        self.batch_size = int(batch_size)
        self.iters = math.ceil(self.data_array.shape[0]/self.batch_size)
        self.index = 0
        self.decrease_k = decrease_k
        
    def Laplacian(X, K=5,sigma=1):
        # X => N x K || HIGH sigma, more leniency on distnace
        X = torch.Tensor(X)
        N_dim = X.shape[0]
        X_expand = X.unsqueeze(1).repeat(1,N_dim,1)
        print("****** Computing Pairwise Distance")
        A_matrix = torch.exp(-1*((X_expand - torch.transpose(X_expand,0,1)).pow(2).sum(dim=2))/(2*sigma**2))-torch.eye(N_dim)
        D_matrix = torch.diag(A_matrix.sum(dim=1)**(-0.5))
        L = D_matrix@A_matrix@D_matrix
        print("****** Computing Eigenvector")
        eig_vals, eig_vecs = eigh(L, eigvals=(N_dim-K,N_dim-1))
        eig_vecs = torch.Tensor(eig_vecs)
        Y_matrix = (eig_vecs/(eig_vecs.pow(2).sum(dim=1).pow(0.5).unsqueeze(dim=1)))
        return Y_matrix.numpy()
        
    def init_cent(self,data_array, k):
        temp_data = data_array.clone()
        num_point = temp_data.shape[0]
        num_dim = temp_data.shape[1]

        centroid = torch.zeros(k, num_dim)
        cent_idx = Tensor(np.random.choice(range(num_point), 1)).long()
        centroid[0,:] = temp_data[cent_idx,:]
        temp_data = temp_data[torch.arange(num_point) != cent_idx]
        num_point -= 1
        for k_i in range(0,k-1,1):
            distance = torch.sqrt(torch.pow(centroid[0:k_i+1,:].unsqueeze(0).repeat(num_point, 1, 1)- 
                             temp_data.unsqueeze(1).repeat(1, k_i+1, 1), 2).sum(dim=-1))
            distance = distance.sum(dim=-1)
            val, idx = torch.max(distance, dim=0)
            centroid[k_i+1,:] = temp_data[idx,:]
            temp_data = temp_data[torch.arange(num_point) != idx]
            num_point -= 1

        return centroid
    
    def get_data(self,index):
        return self.data[index:index+self.batch_size,...]

    def run(self):
        # UPDATE CLUSTEROID BASED ON EACH DATA POINT
        for e in range(self.epochs):
            t = trange(self.iters)
            start = self.cent.clone()
            for i in t: # RUN ALL BATCH
                dt = self.get_data(self.index)
                self.index += self.batch_size
                if CUDA and self.all_cuda==False:
                    dt = dt.cuda(self.device)  
                self.step(dt)
                
                t.set_description("[epoch:%s\t iter:%s] \t k:%s\t distance:%.3f" % (e, i, self.k, self.distance))
            self.index=0
            
        potential = 0
        # FIND NEAREST CLUSTEROID FOR ALL POINTS
        for i in trange(self.iters):
            dt = self.get_data(self.index)
            self.index += self.batch_size
            if CUDA and self.all_cuda==False:
                dt = dt.cuda(self.device)
            if i == 0:
                self.idx, dist_val = self.calc_idx(dt)
                potential += dist_val.cpu().sum()
            else:
                self.idx, dist_val = self.calc_idx(dt)
                self.idx = torch.cat([self.idx, self.idx], dim=-1)
        self.index=0
        
        # RELEASE IDX FROM GPU
        self.idx = self.idx.cpu().numpy()
        return potential

    def step(self, dt):
        idx, val = self.calc_idx(dt)
        self.new_c(idx, dt)

    def calc_distance(self, dt):
        bs = dt.size()[0]
        distance = torch.pow(self.cent.unsqueeze(0).repeat(bs, 1, 1) - dt.unsqueeze(1).repeat(1, self.k, 1), 2).mean(dim=-1) # n_points in batch x k
        return distance

    def calc_idx(self, dt):
        distance = self.calc_distance(dt)
        self.distance = distance.mean().item() # scalar
        val, idx = torch.min(distance, dim=-1) # argmin ==> idx
        return idx, val

    def new_c(self, idx, dt):
        if CUDA:
            z = torch.zeros(self.k, self.dim).cuda(self.device)
            o = torch.zeros(self.k).cuda(self.device)
            ones = torch.ones(dt.size()[0]).cuda(self.device)
        else:
            z = torch.zeros(self.k, self.dim)
            o = torch.zeros(self.k)
            ones = torch.ones(dt.size()[0]) 
            
        ct = o.index_add(0, idx, ones) # adds how many points are in that cluster
        #### slice to remove empty sum (no more such centroid)
        slice_ = (ct > 0) # #########(ct > 0)

        if self.decrease_k == True:
            cent_sum = z.index_add(0, idx, dt)[slice_.view(-1, 1).repeat(1,self.dim)].view(-1, self.dim)
            ct = ct[slice_].view(-1, 1)
            self.cent = cent_sum / ct # new centroid after splicing and dividing by number of points in that cluster
            self.k = self.cent.size()[0] # update number of centroids we have
        else:
            cent_sum = z.index_add(0, idx, dt)[slice_.view(-1, 1).repeat(1,self.dim)].view(-1, self.dim)
            ct = ct[slice_].view(-1, 1)
            self.cent[slice_] = cent_sum/ct


def row_norms(X, squared=False):
    if sparse.issparse(X):
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)
        norms = csr_row_norms(X)
    else:
        norms = np.einsum('ij,ij->i', X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(np.isclose(out.take(-1, axis=axis), expected, rtol=rtol,
                             atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out

def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers




