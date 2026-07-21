from __future__ import print_function

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# https://scikit-learn.org/stable/modules/decomposition.html#pca

import numpy as np
from numpy import linalg as LA

from sklearn.decomposition import PCA
import sklearn.datasets as datasets
def testiris():
    print('>>>> U S I N G  I R I S  D A T A <<<<<\n')
    # Get iris data
    iris = datasets.load_iris()
    X = iris.data
    '''
    #print('Iris data: X=\n', X)
    labels = iris.target
    classlabels = np.unique(iris.target)
    classes = iris.target_names
    featname = iris.feature_names
    print('Feature names: ', featname)
    

    Y = iris.target
    #X = X[:,[2,3]]   # Take only features 3 and 4
    # Subset 50 versicolor vs 50 virginica
    X = X[50:,]     # array index starts counting at zero in Python !
    Y = Y[50:,]
    X2 = X; Y2 = Y
    class1 = classes[1]
    class2 = classes[2]
    '''

    '''
    feat1 = 2   # feature #3
    feat2 = 3   # feature #4
    #X = X[:, [feat1, feat2]]   # Take only features 3 and 4
    X = X[[0, 9, 52, 56, 104, 148], :] # Take six random samples
    '''
    
    
    '''
    [2], p.38: Use class 3 (setosa (Table 4.1, p.39 in [2]) as 'normal' data)
    '''
    X = X[:50,] # setosa only
    #X = X[51:100,] # virginica only
    #X = X[101:150,] # versicolor only
    #X = X[[0,6,9,21,29],:]  # only a few patterns
    print('Iris: Using setosa only\n')
    return X

# Python numpy.linalg.eig does not sort the eigenvalues and eigenvectors
def eigen(A):
    eigenValues, eigenVectors = LA.eig(A)
    idx = np.argsort(eigenValues)
    idx = idx[::-1] # Invert from ascending to descending
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)



X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
X = testiris()


def show_result(pca):
    print('pca.svd_solver=', pca.svd_solver)  
    print('pca.n_components_=', pca.n_components_)  
    print('pca.explained_variance_ratio_=\n', pca.explained_variance_ratio_)  
    print('pca.singular_values_=\n', pca.singular_values_)
    print('pca.mean_=\n', pca.mean_)
    print('pca.noise_variance_=', pca.noise_variance_)
    #print('\n')  

def show_eig(X):
    n_samples = X.shape[0]
    # We center the data and compute the sample covariance matrix.
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
    print('cov_matrix=\n', cov_matrix)    
    eigenvalues = pca.explained_variance_
    for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
        print('eigenvector=', eigenvector)    
        print('phi\'*S*phi=', np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
        print('eigenvalue=', eigenvalue)    

'''
Michael E. Tipping and Christopher M. Bishop
Mixtures of probabilistic principal component analysers, Neural Computation 11(2),
pp 443-482. MIT Press., 1999
This version: June 26, 2006
http://www.miketipping.com/papers.htm
mail@miketipping.com
'''
def PPCA(T):
    # T is data matrix of the n d-dimensional observed samples
    n_samples, d = T.shape
    mu = np.mean(T, axis=0)
    print('mu=', mu)
    # center the data and compute the sample covariance matrix
    T_centered = T - mu
    S = np.dot(T_centered.T, T_centered) / n_samples

    pca = PCA(n_components='mle', svd_solver='full')
    pca.fit(T)     
    sigma2noise = pca.noise_variance_


    print('pca.noise_variance_=', sigma2noise)            

    U = eigvec = pca.components_
    U = U.T  # U is dxq matrix
    Lambda = eigval = pca.explained_variance_
    q = numeigval = pca.n_components_
    print('U.shape=', U.shape, 'U=eigenvectors=\n', eigvec)
    #quit()
    #print('phi\'*S*phi=', np.dot(eigvec.T, np.dot(S, eigvec)))
    print('q=', numeigval, 'of', d, 'eigvals=', eigval)    
    
    sEye = sigma2noise*np.eye(q)
    L = np.sqrt(np.diag(Lambda) - sEye)
    W = np.dot(U, L)
    print('L=\n', L, '\nW.shape=', W.shape, 'W=\n', W)
    
    WWT = np.dot(W, W.T)
    print('WW\'=\n', WWT)
    
    WTW = np.dot(W.T, W)
    print('W\'W=\n', WTW)
    # Model covariance eq. (7)
    sEye_d =  sigma2noise*np.eye(d)
    C = WWT + sEye_d
    print('C=\n', C)
    Cinv = LA.inv(C)
    print('Cinv=\n', Cinv)
    
    Weigval, Weigvec = eigen(WTW)    # Eq. (62)
    R = Weigvec.T
    print('R=\n', R)
    W = np.dot(W, R)
    print('W rotated=\n', W)
    
    # return the three model parameters of the PPCA
    return mu, W, sigma2noise, C, Cinv
    

'''
pca = PCA(n_components=2)
pca.fit(X)
show_eig(X)
#Y = pca.transform(X)
#print('Y=', Y, '\n')
show_result(pca)

pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)                 
show_eig(X)
#Y = pca.transform(X)
#print('Y=', Y, '\n')
show_result(pca)

pca = PCA(n_components=1, svd_solver='arpack')
pca.fit(X)
show_eig(X)
#Y = pca.transform(X)
#print('Y=', Y, '\n')
show_result(pca)
'''

'''
    When n_components is set to 'mle' or a number between 0 and 1
    (with svd_solver == 'full') this number is estimated from input data.
'''
pca = PCA(n_components='mle', svd_solver='full')
print('Data matrix shape=', X.shape)
pca.fit(X)                 
#show_eig(X)
#Y = pca.transform(X)
#print('Y=', Y, '\n')
show_result(pca)


mu, W, sigma2noise, C, Cinv = PPCA(X)
