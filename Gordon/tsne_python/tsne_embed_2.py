import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm, tqdm_notebook
# **********************************************************************
# 1) Averages transition probability across each corresponding points rather than the entire data
# 2) Does not use Nelder-Mead simplex algorthm to optimize but instead uses ADAM
# 3) Starts form a local optimization froom a weighted average of points as Gordon Berman did
# **********************************************************************

def Hbeta(D=torch.Tensor([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
        D - (N_x, )
        P - (N_x, )
        H - Scalar
    """
    # Compute P-row and corresponding perplexity
    P = torch.exp(-torch.Tensor(D) * beta) # ()
    sumP = sum(P)
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=torch.Tensor([]), Y=torch.Tensor([]), tol=0.0001, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
        X - (N_x X D)
        Y - (N_y X D)
    """
    # Initialize some variables
    print("Computing pairwise distances...")
    (n_x, d) = X.shape
    (n_y, d) = Y.shape
    sum_X = torch.sum(torch.pow(X,2), 1) # (N_x, )
    sum_Y = torch.sum(torch.pow(Y,2), 1) # (N_y, )
    D = torch.add(torch.add(-2 * torch.mm(Y, X.t()), sum_X).t(), sum_Y).t() # (N_y, N_x)
    P = torch.zeros((n_y, n_x)) # (N_y, N_x)
    beta = torch.ones((n_y, 1))
    logU = np.log(perplexity)
    # Loop over all datapoints
    for i in range(n_y):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n_y))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        # Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n_y]))] # DISTANCE BETWEEN POINT i AND THE REST
        Di = D[i, :]
        (H, thisP) = Hbeta(Di, beta[i]) # thisP - (N_x,)

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = torch.Tensor(beta[i])
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = torch.Tensor(beta[i])
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        # P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
        P[i, :] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % torch.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne_embed(X=torch.Tensor([]), Xp=torch.Tensor([]), Y=torch.Tensor([]), no_dims=2, initial_dims=50, 
               perplexity=30.0, max_iter=1000, lr=0.1, plot=False):
    # INIT VARIABLES
    (n_x, d) = X.shape
    (n_y, d) = Y.shape
    dY = torch.zeros((n_y, no_dims))
    iY = torch.zeros((n_y, no_dims))
    gains = torch.ones((n_y, no_dims))
    sum_Xp = torch.sum(torch.pow(Xp,2), 1)
    # PCA
    X = torch.Tensor(pca(X.numpy(), initial_dims).real)
    Y = torch.Tensor(pca(Y.numpy(), initial_dims).real)
    # TRANSITION P
    P = x2p(X, Y, perplexity=perplexity) #(N_y, N_x)
#     P = P / torch.sum(P)
    P = P / P.sum(dim=1, keepdim=True)
    P[P<0] = 1e-12
    # START Yp FROM THE WEIGHTED AVERAGE OF POINTS
    Yp = torch.mm(P, Xp)
    Yp.requires_grad_()
#     Yp = torch.rand(n_y, no_dims, requires_grad=True)
    # OPTIMIZE
    opt = optim.Adam([Yp], lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    for epoch in tqdm_notebook(range(max_iter)):
        # TRANSITION Q
        sum_Yp = torch.sum(torch.pow(Yp,2), 1)
        num = torch.Tensor(-2. * torch.mm(Xp, Yp.t()))
        num = 1. / (1. + torch.add(torch.add(num.t(), sum_Xp).t(), sum_Yp)).t() # (N_y, N_x)
#         Q = num / torch.sum(num) # (N_y, N_x)
        Q = num / num.sum(dim=1, keepdim=True)# (N_y, N_x)
        Q[Q<0] = 1e-12
        def closure():
            # OPTIMIZE
            opt.zero_grad()
            # loss=torch.pow(P-Q, 2).sum() # MSE Loss
            loss = torch.sum(P * torch.log(P / Q)) # KL Divergence Loss
            loss.backward()
            return loss
        opt.step(closure)
        # PLOT
        if plot and epoch % 100 == 0:
            # loss=torch.pow(P-Q, 2).sum() # MSE Loss
            loss = torch.sum(P * torch.log(P / Q)) # KL Divergence Loss
            tqdm.write("Epoch {} - Loss: {}".format(epoch, round(loss.item(),5)))
            plt.scatter(Xp[:,0].numpy(), Xp[:,1].numpy())
            plt.scatter(Yp[:,0].detach().numpy(), Yp[:,1].detach().numpy())
            plt.show()
    Yp = Yp.detach()
    P = P.detach()
    Q = Q.detach()
    return Yp, P, Q