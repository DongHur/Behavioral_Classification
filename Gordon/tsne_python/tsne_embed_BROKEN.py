import numpy as np
import pylab


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
        D - (N_x, )
        P - (N_x, )
        H - Scalar
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta) # ()
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), Y=np.array([]), tol=1e-5, perplexity=30.0):
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
    sum_X = np.sum(np.square(X), 1) # (N_x, )
    sum_Y = np.sum(np.square(Y), 1) # (N_y, )
    D = np.add(np.add(-2 * np.dot(Y, X.T), sum_X).T, sum_Y).T # (N_y, N_x)
    P = np.zeros((n_y, n_x)) # (N_y, N_x)
    beta = np.ones((n_y, 1))
    logU = np.log(perplexity)
    #************CHECKED UP TO HERE **********
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
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
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
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
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


def tsne_embed(X=np.array([]), Xp=np.array([]), Y=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000):
    """
        t-SNE embed data points Y to Y_prime based on the X and X_prime
    """

    # Initialize variables
    X = pca(X, initial_dims).real
    Y = pca(Y, initial_dims).real
    (n_x, d) = X.shape
    (n_y, d) = Y.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Yp = np.random.randn(n_y, no_dims)
    dY = np.zeros((n_y, no_dims))
    iY = np.zeros((n_y, no_dims))
    gains = np.ones((n_y, no_dims))

    # Compute P-values
    P = x2p(X, Y, tol=1e-5, perplexity=perplexity) # (N_y, N_x)
    # P = P + np.transpose(P) # TAKING THIS OUT BECAUSE P MATRIX IS NO LONGER SQUARE
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Xp = np.sum(np.square(Xp), 1)
        sum_Yp = np.sum(np.square(Yp), 1)
        num = torch.Tensor(-2. * np.dot(Xp, Yp.T), requires_grad=True)
        num = 1. / (1. + np.add(np.add(num.T, sum_Xp).T, sum_Yp)).T # (N_y, N_x)
        # num[range(n), range(n)] = 0. # NO VARIABLE IS WITH ITSELF
        Q = num / np.sum(num) # (N_y, N_x)
        Q = np.maximum(Q, 1e-12) 
        
        Q.detach()
        
#         # Compute gradient
        PQ = P - Q # (N_y, N_x) 
        for i in range(n_y):
#             dY[i, :] = np.sum(np.tile(PQ[i, :] * num[i, :], (no_dims, 1)).T * (Yp[i, :] - Xp), 0)
            # YOU DON'T WANT TO MINIMIZE THE DISTNAT BETWEEN Yp and all the Xp. BUT BY SUMMING OVER ALL THE OF FIRST
            # DIMENSION YOU ARE LOSING SOME OF THE DIFFERENCE IN TRANSITION PROBILITY WITH THE TRAINING POINT
            dY[i, :] = np.sum(np.tile(PQ[i, :], (no_dims, 1)).T, 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Yp = Yp + iY
        Yp = Yp - np.tile(np.mean(Yp, 0), (n_y, 1))
        
        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
            plt.scatter(Yp[:,0], Yp[:,1])
            plt.scatter(Xp[:,0], Xp[:,1])
            plt.show()
            plt.plot(P[0,:])
            plt.plot(Q[0,:])
            plt.show()

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Yp, P, Q
