# FIND EMBEDDINGS FOR THE REST OF THE DATA POINTS
class Embeddings():
    def __init__(self, X, X_prime, Y):
        # X - (N x D); trained feature data
        # X_prime - (N_x x D); embedded trained data
        # Y - (N_y x 2); untrained feature data
        self.X = Tensor(X)
        self.X_prime = Tensor(X_prime)
        self.Y = Tensor(Y)
        
        self.N_x = X.shape[0]
        self.N_y = Y.shape[0]
        self.D = X.shape[1]
        
    def _compute_gauss_prob(self, ds, sigma):
        p = torch.exp(-.5*ds**2/sigma**2)
        p = p/p.sum()
        idx = p>0;
        H = (-p[idx]*torch.log(p[idx])/np.log(2)).sum()
        P = 2**H;
        return P
    def _compute_t_prob(self, ds):
        # ds - (Nx x Ny)
        q = (1+ds**2)**(-1)
        q = q/q.sum(dim=0, keepdim=True)
        return q
    
    def KL_divergence(self, data, data2): 
        log_data = torch.log(data)
        log_data2 = torch.log(data2)
        log_data[torch.isnan(log_data)|torch.isinf(log_data)] = 0
        log_data2[torch.isnan(log_data2)|torch.isinf(log_data2)] = 0
        test = -1.0*(data*log_data)
        entropies = -1.0*(data*log_data).sum(dim=1)
        D = -1.0*data@log_data2.t()
        D = D-entropies.unsqueeze(dim=-1)
        D = D/np.log(2);
        return D
    def find_sigma(self, D2_i, max_neighbors=5, perplexity = 4, tol = 1e-5):
        # MAX_NEIGHBORS MUST BE GREATER THAN THE PERPLEXITY
        high_guess = D2_i.max()
        low_guess = 1e-10
        sigma = (high_guess+low_guess)/2
        # SORT TO FIND IDX OF NEAREST EMBEDDING POINTS FROM IMPT SAMPLING
        sort_values, sort_idx = torch.sort(D2_i, dim=0)
        D2_i_sort = D2_i[sort_idx[0:max_neighbors]]
        P_entropy = self._compute_gauss_prob(D2_i_sort, sigma)

        if abs(P_entropy-perplexity) < tol:
            test = False
        else:
            test = True
        while test:
            if P_entropy > perplexity:
                high_guess = sigma
            else:
                low_guess = sigma
            sigma = .5*(high_guess + low_guess)
            P_entropy = self._compute_gauss_prob(D2_i_sort, sigma)
            if abs(P_entropy-perplexity) < tol:
                test = False
        # RECOMPUTE p
        p = torch.exp(-.5*D2_i**2/sigma**2)
        p = p/p.sum()
        return sigma, p, sort_idx[0:max_neighbors]
    
    def run(self, D2, epoch=100, eta=0.1, alpha=0.0001):
        max_neighbors = 10
        gauss_prob = torch.zeros(self.N_x, self.N_y)
        for i in range(self.N_x):
            sigma, p, sort_idx = self.find_sigma(D2[i,:], max_neighbors=max_neighbors)
            gauss_prob[i,:] = p
        # *************************
        grad_history = np.zeros(epoch)
        # RANDOMLY SAMPLE FOR Y INITALLY
        y_mean = torch.zeros(self.N_y, 2)
        Y_prime = torch.normal(y_mean, 1e-4)
        Y_prime_prev = Y_prime
        for i in range(epoch):
            # COMPUTE q (WHICH IS BETWEEN LOWER DIMENSIONS)
            ds_embed = self.KL_divergence(self.X_prime, Y_prime)
            t_prob = self._compute_t_prob(ds_embed)
            # COMPUTE GRADIENT
            # *** do i make this negative? ***
            y_diff = -(self.X_prime[np.newaxis,:,:]-Y_prime[:,np.newaxis,:]).transpose(0,1)
            prob_diff = (gauss_prob-t_prob).unsqueeze(-1)
            t_dist_diff = ((1+ds_embed**2)**-1).unsqueeze(-1)
            
            grad = 4*(prob_diff*y_diff*t_dist_diff).sum(dim=0)
            grad_history[i] = grad.sum()
            # TAKE Y DOWN GRADIENT
            Y_new = Y_prime + eta*grad + alpha*(Y_prime-Y_prime_prev)
            Y_prime_prev = Y_prime
            Y_prime = Y_new
            
        plt.plot(grad_history)
        plt.show()
        return Y_prime
        
Embed = Embeddings(X=training_data, X_prime=X_prime, Y=pwr)
D2 = Embed.KL_divergence(Embed.X, Embed.Y)
Y_prime = Embed.run(D2)