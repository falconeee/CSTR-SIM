#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Thu Aug  9 11:52:29 2018

@author: thomas
"""

"""
========================================
[1] Reconstruction-based contribution for process monitoring,
    Carlos F. Alcala, S. Joe Qin
    Automatica 45 (2009), pp. 1593-1600
    
[2] Fault Detection and Diagnosis in Industrial Systems,
    Chiang, L.H., Russell, E.L., Braatz, R.D, 2001

[3] Assessment of T2- and Q-statistics for detecting additive and multiplicative
    faults in multivariate statistical process monitoring,
    Kai Zhang, Steven X. Ding, Yuri A.W. Shardt, Zhiwen Chen, Kaixiang Peng,
    Journal of the Franklin Institute 354 (2017) 668–688
    
[4] Some theorems on quadratic forms applied in the study of analysis of variance problems,
    Box, 1954, https://projecteuclid.org/euclid.aoms/1177728786
    
[5] Data-driven Design of Fault Diagnosis and Fault-tolerant Control Systems,
    Steven X. Ding, Springer, 2014
    
[6] Control Procedures for Residuals Associated With Principal Component Analysis,
    Jackson,Mudholkar, Technometrics, 1979
    
[7] Press, W.H.; S.A. Teukolsky; W.T. Vetterling; B.P. Flannery (1992) [1988].
    Numerical Recipes in C: The Art of Scientific Computing (2nd ed.).
    Cambridge UK: Cambridge University Press.
    https://en.wikipedia.org/wiki/Confidence_region

[8] Reconstruction-Based Fault Identification Using a Combined Index,
    H. Henry Yue, S. Joe Qin
    Ind. Eng. Chem. Res. 2001, 40, 4403-4414
    
[9] Statistical process monitoring: basics and beyond
    S. Joe Qin,
    J. Chemometrics 2003; 17: 480–502 , DOI: 10.1002/cem.800

[10] Journal of Process Control, Volume 20, Issue 10, Dec. 2010, Pages 1198-1206
    A Branch and Bound Method for Isolation of Faulty Variables through Missing Variable Analysis
    Vinay Kariwala , Pabara-Ebiere Odiowei , Yi Cao and Tao Chen


========================================

Didactical presentation of PCA
"""

import sklearn.datasets as datasets
import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.stats import chi2
from plot_2_D_statistics import plot_2_D_statistics


# ddof=1 ==> divide by (n-1) --- ddof=0 ==> divide by n
ddof_std = 0    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.std.html#numpy.std


#print(__doc__)




# Angle and its cosine between two vectors
def angle_between(u,v):
	cosangle = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v) # -> cosine of the angle
	angle = np.arccos(np.clip(cosangle, -1, 1)) # angle in radians
	return angle, cosangle

# Python numpy.linalg.eig does not sort the eigenvalues and eigenvectors
def eigen(A):
    eigenValues, eigenVectors = LA.eig(A)
    idx = np.argsort(eigenValues)
    idx = idx[::-1] # Invert from ascending to descending
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)

def SPE(x, Ctil):
    return np.dot(np.dot(x.T,Ctil),x)

def T2(x, D):
    return np.dot(np.dot(x.T,D),x)

def combined_phi(x, PHI):
    return np.dot(np.dot(x.T,PHI),x)

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



def Index_and_individual_contribs(X, M, verbose=False):    # [1] Eq. (8),(9),(10)
    Msqrt = sqrtm(M)
    #print('M=\n', M, '\nMsqrt=\n', Msqrt, '\nMsqrt^2=', np.dot(Msqrt,Msqrt))
    #print('X.shape=', X.shape, 'X=\n', X)

    individual_contribs = np.dot(X,Msqrt)**2
    if verbose:
        print('individual_contribs=', individual_contribs)
    Index = np.sum(individual_contribs, axis=1)
    if verbose:
        print('Index=', Index)
    return individual_contribs, Index





def train_reconstruction_based(training):
    
    #  [1]  3. Contribution by reconstruction
    print('\n\n --- [1]  3. Contribution by reconstruction ---\n\n')
    
    M_c_SPE = training.get('M_c_SPE')

    
    cii = np.diag(M_c_SPE)
    M_RBC_SPE = M_c_SPE / cii[None,:] # Eq.(28)  # division of corresponding column elements
    ######M_RBC_SPE = c_SPE / cii[None,:] # Eq.(28)  # division of corresponding column elements
    print('\n\ncii=', cii, '\nM_RBC_SPE\n', M_RBC_SPE)
    training.update({'M_RBC_SPE': M_RBC_SPE})


    '''
    Xtil = np.dot(X, M_c_SPE) # Eq. (4)
    c_SPE = Xtil**2 # Eq. (10), (12) element-wise square
    print('\nc_SPE\n', c_SPE)

    M_RBC_T2 = (np.dot(X, D)**2) / ((np.diag(D))[None,:]) # Eq.(29)
    print('M_RBC_T2=\n', M_RBC_T2)
    
    M_RBC_phi = (np.dot(X, PHI)**2) / ((np.diag(PHI))[None,:]) # Eq.(30)
    print('M_RBC_phi=\n', M_RBC_phi)
    
    training.update({'M_RBC_T2': M_RBC_T2})
    training.update({'M_RBC_phi': M_RBC_phi})
    '''

    #  [1]  3.2. Control limits for the reconstruction-based contributions
    print('\n\n [1]  3.2. Control limits for the reconstruction-based contributions ---\n\n')

    h_i_Index = 1.0
    alpha = 0.05
    chi = chi2.ppf(1-alpha, h_i_Index)
    
    M = M_c_SPE
    MSM = np.dot(np.dot(M, S), M)
    RBC_g_SPE = np.diag(MSM)  / np.diag(M)   # Eq. 25
    RBC_gamma2_SPE = RBC_g_SPE * chi
    print('RBC_gamma2_SPE=', RBC_gamma2_SPE)
    
    M = M_c_T2
    MSM = np.dot(np.dot(M, S), M)
    RBC_g_T2 = np.diag(MSM)  / np.diag(M)
    RBC_gamma2_T2 = RBC_g_T2 * chi
    print('RBC_gamma2_T2=', RBC_gamma2_T2)
    
    M = M_c_combined
    MSM = np.dot(np.dot(M, S), M)
    RBC_g_combined = np.diag(MSM)  / np.diag(M)
    RBC_gamma2_combined = RBC_g_combined * chi
    print('RBC_gamma2_combined=', RBC_gamma2_combined)

    training.update({'RBC_gamma2_SPE': RBC_gamma2_SPE})
    training.update({'RBC_gamma2_T2': RBC_gamma2_T2})
    training.update({'RBC_gamma2_combined': RBC_gamma2_combined})
    
    
    
    
    M_RBC_SPE = training.get('M_RBC_SPE')
    RBC_gamma2_SPE = training.get('RBC_gamma2_SPE')

    #print('training_data.shape=', training_data.shape, 'M_c_SPE.shape=', M_c_SPE.shape, 'M_RBC_SPE.shape=', M_RBC_SPE.shape, 'fault_data.shape=', fault_data.shape ) 
    SPE_RBC_X_training, SPE_RBC_i_X_training = Index( training_data, M_RBC_SPE)

    SPE_RBC_fault_contrib_X_training = (SPE_RBC_i_X_training > RBC_gamma2_SPE).astype(int)
    #print('SPE_RBC_fault_contrib_X_training=\n', SPE_RBC_fault_contrib_X_training)

    tp, fa = evaluate(SPE_RBC_fault_contrib_X_training, training_labels)
    print('Training data: tp=', tp, 'fa=', fa, 'n=', training_labels.shape[0])


    SPE_RBC_X_test, SPE_RBC_i_X_test = Index( fault_data, M_RBC_SPE)

    SPE_RBC_fault_contrib_X_test = (SPE_RBC_i_X_test > RBC_gamma2_SPE).astype(int)
    #print('SPE_RBC_fault_contrib_X_test=\n', SPE_RBC_fault_contrib_X_test)

    tp, fa = evaluate(SPE_RBC_fault_contrib_X_test, fault_labels)
    print('Test data: tp=', tp, '=', 100.0*tp/fault_labels.shape[0], '%', 'fa=', fa, 'n=', fault_labels.shape[0])


    print('\n\nRECONSTRUCTION-BASED DIAGNOSIS\n')
    # S P E  R E C O N S T R U C T I O N - B A S E D


    def evaluate(Yhat, Y):
        n, d = training_labels.shape
        num_correct_diag = 0
        num_false_alarm = 0
        for j in range(n):
            diagnosed = True
            false_alarm = False
            for i in range(d):
                if Y[j][i]==1:
                    if Yhat[j][i]!=1:
                        diagnosed = False
                else:
                    if Yhat[j][i]==1:
                        false_alarm = True
            if diagnosed:
                num_correct_diag += 1
            if false_alarm:
                num_false_alarm += 1
            #print('Pattern', j+1, 'of', n, 'Diagnosed=', Yhat[j], 'True=', Y[j], 'diagnosis=', diagnosed, 'false_alarm=', false_alarm)
        return num_correct_diag, num_false_alarm



    M_c_SPE = training.get('M_c_SPE')
    SPE_c_X_training, SPE_c_i_X_training = Index( training_data, M_c_SPE)

    return training



def gen_train(Xtrain, Xtest, num_principal_components=None):
    #print('gen_train> Dataset before standardization=\n', X)
    training = {} # dictionary
    training.update({'num_principal_components': num_principal_components})
    X = Xtrain
    m, n = X.shape # m samples, n variables, data matrix has dimension (m,n) 
    print ('Number of samples = ', m, 'Number of variables = ', n)
    meanX = np.mean(X, axis=0)
    stdX = X.std(axis=0, ddof=ddof_std)
    print('Dataset statistic:\n Mean=', meanX, '\nStandard deviation=\n', stdX)
    minX = X.min(axis=0)
    maxX = X.max(axis=0)
    print('Dataset statistic:\nMin=', minX, '\nMax=', maxX )
    Xmean = X - meanX
    #print('Dataset X=\n', X, '\nDataset centralized Xmean=\n', Xmean)
    Xtrain_norm = Xmean / stdX
    #print('Training dataset standadized Xtrain_norm=\n', Xtrain_norm)

    Xtest_norm = (Xtest - meanX) / stdX
    print('Test dataset standadized Xtest_norm=\n', Xtest_norm)


    training.update({'num_samples': m})
    training.update({'num_variables': n})
    training.update({'Xtrain': Xtrain})
    training.update({'Xtest': Xtest})
    training.update({'meanX': meanX})
    training.update({'stdX': stdX})
    training.update({'minX': minX})
    training.update({'maxX': maxX})
    training.update({'Xtrain_norm': Xtrain_norm})
    training.update({'Xtest_norm': Xtest_norm})
    
    S = np.cov(X, ddof=1, rowvar=False) # Observations (samples) are the rows Eq. (2)
    Lambda, Phi = eigen(S)
    print('Covariance Matrix of Dataset S=\n', S)
    print('Eigenvectors of Covariance Matrix of Dataset (loadings): Phi=\n', Phi)
    print('Eigenvalues of Covariance Matrix of Dataset: Lambda=\n', Lambda)
    '''
    print('Phi * Phi\'=\n', np.dot(Phi, Phi.T))
    print('S * Phi=\n', np.dot(S, Phi))
    print('\nPhi * LAMBDA=\n', np.dot(Phi, np.diag(Lambda)))
    '''
    training.update({'S': S})
    training.update({'Lambda': Lambda})
    training.update({'Phi': Phi})
    
    
    #l = n-2 # Number of principal components
    l = num_principal_components
    P = Phi[:, :l]
    print('First ', l, 'columns of Loading matrix Phat=\n', P)
    L = np.diag(Lambda[:l])
    print('Diagonal matrix of first ', l, 'eigenvalues L=\n', L)
    
    # T2 statistic
    LInv = np.diag((1.0/Lambda[:l]))
    print('Diagonal matrix of the inverse of the first ', l, 'eigenvalues L=\n', LInv)
    D = np.dot(np.dot(P, LInv), P.T)
    print('Matrix D = PL^-1P''=\n', D)
    M_c_T2 = D 
    print('T2 whitening subspace projection matrix M_T2=\n', M_c_T2)

    
    Ptil = Phi[:, l:]
    print('Last ', n-l, 'columns of Loading matrix Phat=\n', Ptil)
    Ltil = np.diag(Lambda[l:])
    print('Diagonal matrix of last ', n-l, 'eigenvalues Ltil=\n', Ltil)
    training.update({'P': P})
    training.update({'Ptil': Ptil})
    training.update({'L': L})
    training.update({'Ltil': Ltil})
    
    Shat = np.dot(np.dot(P, L), P.T)
    print('Shat=', Shat)
    Stil = np.dot(np.dot(Ptil, Ltil), Ptil.T)
    print('Stil=', Stil)
    print('Verify (should be zero): S=Shat+Stil\n', (Shat+Stil)-S) # Eq. (3)
    
    C = np.dot(P, P.T)
    Ctil = np.dot(Ptil, Ptil.T)
    M_c_SPE = Ctil
    print('Principal component subspace (PCS) projection matrix C=\n', C)
    print('Residual subspace (RS) projection matrix M_c_SPE=Ctil=\n', Ctil)
    
    
    # 2.2.1. Eq. (4) Squared prediction error, SPE
    
    theta1 = np.sum(Lambda[l:])
    theta2 = np.sum(np.square(Lambda[l:]))
    g_SPE = theta2 / theta1
    h_SPE = theta1**2 / theta2
    alpha = 0.05
    deltasqr = g_SPE * chi2.ppf(1-alpha, h_SPE)
    print('theta1=', theta1, 'theta2=', theta2, 'g_SPE=', g_SPE,
          'h_SPE=', h_SPE, 'delta=', np.sqrt(deltasqr), 'deltasqr=', deltasqr)
    
    training.update({'deltasqr': deltasqr})
    training.update({'M_c_SPE': M_c_SPE})

    '''
     T^2 statistic
     [1], Eq. (5)
     [2], Eq. 2.10, p.22; Eq. 2.11, Eq. 4.14, p.43 as an alternative
    '''
    from scipy.stats import f as F_distribution
    # T2 control limit
    tau2 = (l*(m-1)*(m+1)/(m*(m-l)))*F_distribution.ppf(q=1-alpha, dfn=l, dfd=m-l)
    print('Braatz et al.: l=a=', l, 'm=n=', m, 'tau2 Braatz=', tau2)
    
    tau2 = chi2.ppf(1-alpha, l) # Qin
    print('tau2 Qin=', tau2)
    
    training.update({'tau2': tau2})
    training.update({'M_c_T2': M_c_T2})
    
    # Combined index: [1] Eq. (6)
    PHI = Ctil/deltasqr + D/tau2
    print('Combined index matrix: PHI=\n', PHI)
    M_c_combined = sqrtm(PHI)
    print('Combined projection matrix M_c_combined=\n', M_c_combined)
    
    # Combined index control limit
    aux1 = 1.0/tau2**2 + theta2/deltasqr**2
    aux2 = 1.0/tau2 + theta1/deltasqr
    g_phi = aux1 / aux2
    h_phi = aux2**2 / aux1
    zeta2 = g_phi * chi2.ppf(1-alpha, h_phi)
    print('g_phi=', g_phi, 'h_phi=', h_phi, 'zeta2=', zeta2)
    
    training.update({'zeta2': zeta2})
    training.update({'M_c_combined': M_c_combined})


    # [10] M2 statistic
    from probabilistic_PCA import PPCA_parameters, PPCA
    n_samples, d, M2_control_limit = PPCA_parameters(Xtrain)
    mu, W, sigma2noise, C, Cinv = PPCA(Xtrain)
    training.update({'M2_control_limit': M2_control_limit})
    M_c_M2 = Cinv
    training.update({'M_c_M2': M_c_M2})
    
    print('\nT R A I N I N G  P A R A M E T E R S  L E A R N E D\n')
    return training



def detect_contrib(training):
    #  [1]  2.3. Fault diagnosis by contribution plots   
    print('\n\n --- [1]  2.3. Fault diagnosis by contribution plots ---\n\n')
    
    Ctil= M_c_SPE = training.get('M_c_SPE')
    X = training.get('Xtrain_norm')

    # SPE contribution
    #print('Ctil=M_c_SPE=\n', Ctil)
    Xtil = np.dot(X, Ctil) # Eq. (4)
    c_SPE = Xtil**2 # Eq. (10), (12) element-wise square, each line
    #                   contains [c_0, c_1, ...] of all the samples x
    SPE = np.sum(c_SPE, axis=1) # Eq. (4) SPE statistic of each sample
    #print('Xtil=\n', Xtil, '\nindividual contribs c_SPE=\n', c_SPE, '\nSPE=', SPE)

    '''
    c_SPE_new, SPE_new = Index_and_individual_contribs(X, Ctil)
    diff = c_SPE - c_SPE_new
    print('SPE diff=', c_SPE - c_SPE_new, '\n norm=', LA.norm(diff),
            '\n', SPE-SPE_new, '\nnorm=', LA.norm(SPE-SPE_new)); quit()
    '''
    
    # T2 contribution
    D = training.get('M_c_T2')
    Dsqrt = sqrtm(D) # cholesky does not work
    c_T2 = np.dot(X, Dsqrt)**2  # Eq. (5)
    T2 = np.sum(c_T2, axis=1)
    #print('c_T2.shape=\n', c_T2.shape, 'individual contribs c_T2=\n', c_T2)

    '''
    c_T2new, T2_new = Index_and_individual_contribs(X, D)    # [1] Eq. (13),(5)
    print('diff=', c_T2 - c_T2new, '\n\nT2=', T2_new, 'T2_new.shape=', T2_new.shape)
    quit()
    '''
    
    # Combined index contribution
    PHI = training.get('M_c_combined')
    c_phi = np.dot(X, sqrtm(PHI))**2
    phi = np.sum(c_phi, axis=1) # Eq. (6)
    '''
    print('c_phi=\n', c_phi, '\nphi=', phi)
    c_phi_new, phi_new = Index_and_individual_contribs(X, PHI)
    print('diff phi norm=', LA.norm(c_phi-c_phi_new), LA.norm(phi-phi_new))
    quit()
    '''

    # M2 contribution
    M_c_M2 = training.get('M_c_M2')
    c_M2, M2 = Index_and_individual_contribs(X, M_c_M2)
    #print('c_M2.shape=\n', c_M2.shape, 'individual contribs c_M2=\n', c_M2, '\nM2=', M2); quit()





def test():

    #from tedata import csvread
    import tedata

    XMV = tedata.XMV
    XMEAS = tedata.XMEAS


    f = '/home/thomas/Dropbox/software/TE/Tennessee_Eastman/TE_process/data/d16_te.dat'
    f = '/home/thomas/Dropbox/software/TE/Tennessee_Eastman/TE_process/data/d16.dat'
    #f = '/home/thomas/Dropbox/software/TE/ORIGINAL_LEAVE_UNCHANGED/TE_process/d19_te.dat'
    delimiter='\t'
    delimiter=' '
    X = tedata.csvreadTE(f, delimiter=delimiter)
    #X, Y, y, ynum, classname = tedata.csvread('./all.csv')


    #X = testiris()
    print('X data matrix=\n', X)




    print('\nDataset before standardization=\n', X)
    meanX = np.mean(X, axis=0)
    stdX = X.std(axis=0, ddof=ddof_std)
    print('Dataset statistic:\n Mean=', meanX, '\nStandard deviation=\n', stdX)
    minX = X.min(axis=0)
    maxX = X.max(axis=0)
    print('Dataset statistic:\nMin=', minX, '\nMax=', maxX )

    Xmean = X - meanX
    #print('Dataset X=\n', X, '\nDataset centralized Xmean=\n', Xmean)
    Xnorm = Xmean / stdX
    #print('Dataset standadized Xnorm=\n', Xnorm)
    X = Xnorm


    training = {} # dictionary
    training.update({'X': X})
    training.update({'meanX': meanX})
    training.update({'stdX': stdX})


    m, n = X.shape # m samples, n variables, data matrix has dimension (m,n) 
    print ('Number of samples = ', m, 'Number of variables = ', n)

    print('Data Matrix X=\n', X)
    S = np.cov(X, rowvar=False) # Observations (samples) are the rows Eq. (2)
    print('Covariance Matrix of Dataset S=\n', S)

    Lambda, Phi = eigen(S)
    print('Eigenvectors of Covariance Matrix of Dataset (loadings): Phi=\n', Phi)
    print('Eigenvalues of Covariance Matrix of Dataset: Lambda=\n', Lambda)
    '''
    print('Phi * Phi\'=\n', np.dot(Phi, Phi.T))
    print('S * Phi=\n', np.dot(S, Phi))
    print('\nPhi * LAMBDA=\n', np.dot(Phi, np.diag(Lambda)))
    '''
    #quit()

    #l = n-2 # Number of principal components
    #l = 2
    l = 3
    P = Phi[:, :l]
    print('First ', l, 'columns of Loading matrix Phat=\n', P)
    L = np.diag(Lambda[:l])
    print('Diagonal matrix of first ', l, 'eigenvalues L=\n', L)

    # T2 statistic
    LInv = np.diag((1.0/Lambda[:l]))
    print('Diagonal matrix of the inverse of the first ', l, 'eigenvalues L=\n', LInv)
    D = np.dot(np.dot(P, LInv), P.T)
    print('Matrix D = PL^-1P''=\n', D)
    M_T2 = np.dot(P, np.diag(1.0/np.sqrt(Lambda[:l])) ) 
    print('T2 whitening subspace projection matrix M_T2=\n', M_T2)



    Ptil = Phi[:, l:]
    print('Last ', n-l, 'columns of Loading matrix Phat=\n', Ptil)
    Ltil = np.diag(Lambda[l:])
    print('Diagonal matrix of last ', n-l, 'eigenvalues Ltil=\n', Ltil)

    Shat = np.dot(np.dot(P, L), P.T)
    print('Shat=', Shat)
    Stil = np.dot(np.dot(Ptil, Ltil), Ptil.T)
    print('Stil=', Stil)
    print('Verify (should be zero): S=Shat+Stil\n', (Shat+Stil)-S) # Eq. (3)

    C = np.dot(P, P.T)
    Ctil = np.dot(Ptil, Ptil.T)
    M_c_SPE = Ctil
    print('Principal component subspace (PCS) projection matrix C=\n', C)
    print('Residual subspace (RS) projection matrix M_c_SPE=Ctil=\n', Ctil)


    # 2.2.1. Eq. (4) Squared prediction error, SPE
    from scipy.stats import chi2

    theta1 = np.sum(Lambda[l:])
    theta2 = np.sum(np.square(Lambda[l:]))
    g_SPE = theta2 / theta1
    h_SPE = theta1**2 / theta2
    alpha = 0.05
    deltasqr = g_SPE * chi2.ppf(1-alpha, h_SPE)
    delta = np.sqrt(deltasqr)
    print('theta1=', theta1, 'theta2=', theta2, 'g_SPE=', g_SPE, 'h_SPE=', h_SPE,
        'deltasqr=', deltasqr)

    # Calculate the SPE for all x of the data matrix X Eq.(4), page 1594
    #SPE = np.sum( np.dot(np.dot(X, Ctil),Ctil.T) * X, axis=1)
    SPE_c_X = np.sum( np.dot(X, Ctil) * X, axis=1) # Eq. (4)
    print('SPE_c_X=', SPE_c_X)

    training.update({'deltasqr': deltasqr})
    training.update({'SPE_c_X': SPE_c_X})

    '''
        Alternative for threshold of Q-statistic
        [2], Eq. 4.22, p. 44 ==> Formula is wrong: lamda^(i), not lambda^(2i)
        Original paper = [6], Eq. 3.4, p. 342
    '''
    theta3 = np.sum(np.square(Lambda[l:])*Lambda[l:])

    from scipy.stats import norm
    calpha = norm.ppf(1-alpha)
    h0 = 1 - (2*theta1*theta3)/(3*theta2**2)
    Qalpha = theta1 * (h0*calpha*np.sqrt(2*theta2)/theta1 + 1 + (theta2*h0*(h0-1)/theta1**2))**(1/h0)
    print('theta1=', theta1, 'theta2=', theta2, 'theta3=', theta3,
        'calpha=', calpha, 'h0=', h0, '\nQalpha=', Qalpha, 'to compare: deltasqr=', deltasqr)


    '''
    T^2 statistic
    [1], Eq. (5)
    [2], Eq. 2.10, p.22; Eq. 2.11, Eq. 4.14, p.43 as an alternative
    '''
    from scipy.stats import f as F_distribution
    # Control limit
    tau2 = (l*(m-1)*(m+1)/(m*(m-l)))*F_distribution.ppf(q=1-alpha, dfn=l, dfd=m-l)
    print('Braatz et al.: l=a=', l, 'm=n=', m, 'tau2 Braatz=', tau2)

    tau2 = chi2.ppf(1-alpha, l) # Qin
    print('tau2 Qin=', tau2)

    # Calculate the T2 for all x of the data matrix X Eq.(4), page 1594
    T2_X = np.sum( np.dot(X, D) * X, axis=1) # Eq. (4)
    print('T2_X=', T2_X)

    # This works not, if individual contributions have to be calculated.
    # In that case:
    #
    verify_test = True
    verify_test = False
    if verify_test:
        Dsqrt = sqrtm(D) # cholesky does not work
        T2_X_indiv = np.dot(X, Dsqrt)
        T2_X_indiv = T2_X_indiv * T2_X_indiv
        print('T2_X_indiv=\n', T2_X_indiv)
        T2_X_indiv = np.sum(T2_X_indiv, axis=1)
        DIFF = T2_X_indiv - T2_X
        print('Verify: DIFF=', DIFF, '\nNorm2(DIFF)=', np.dot(DIFF,DIFF))
        quit()

    training.update({'tau2': tau2})
    training.update({'T2_X': T2_X})



    # Combined index: [1] Eq. (6)
    PHI = Ctil/deltasqr + D/tau2
    print('Combined index matrix: PHI=\n', PHI)
    M_combined = sqrtm(PHI)
    print('Combined projection matrix M_combined=\n', M_combined)



    # Combined index control limit
    aux1 = 1.0/tau2**2 + theta2/deltasqr**2
    aux2 = 1.0/tau2 + theta1/deltasqr
    g_phi = aux1 / aux2
    h_phi = aux2**2 / aux1
    zeta2 = g_phi * chi2.ppf(1-alpha, h_phi)
    print('g_phi=', g_phi, 'h_phi=', h_phi, 'zeta2=', zeta2)

    combined_phi_X = np.sum( np.dot(X, PHI) * X, axis=1) # Eq. (4)
    print('combined_phi_X=', combined_phi_X)

    training.update({'zeta2': zeta2})
    training.update({'combined_phi_X': combined_phi_X})


    NormalConditionSPE = SPE_c_X < deltasqr
    numFaultsSPE = list(NormalConditionSPE).count(False)
    #print('Normal Condition: SPE < deltasqr=', NormalConditionSPE)
    print('# Faults for SPE = ', numFaultsSPE)
    NormalConditionT2 = T2_X < tau2
    numFaultsT2 = list(NormalConditionT2).count(False)
    #print('Normal Condition: T2 < tau2=', NormalConditionT2)
    print('# Faults for T2 = ', numFaultsT2)
    NormalConditionCombined = combined_phi_X < zeta2
    numFaultsCombined = list(NormalConditionCombined).count(False)
    #print('Normal Condition: combined_phi_X < zeta2=', NormalConditionCombined)
    print('# Faults for combined index = ', numFaultsCombined)

    #  [1]  2.3. Fault diagnosis by contribution plots   

    # SPE contribution
    print('n=', n, 'Ctil=\n', Ctil)
    Xtil = np.dot(X, Ctil) 
    c_SPE = Xtil**2 # element-wise square, each line contains [c_0, c_1, ...] of the sample x
    print('Xtil=\n', Xtil, '\nc_SPE=\n', c_SPE)

    # T2 contribution
    Dsqrt = sqrtm(D)
    c_T2 = np.dot(X, Dsqrt)**2
    print('c_T2=\n', c_T2)

    # Combined index contribution
    c_phi = np.dot(X, sqrtm(PHI))**2
    print('c_phi=\n', c_phi)


    #  [1]  3. Contribution by reconstruction
    cii = np.diag(Ctil)
    RBC_SPE = c_SPE / cii[None,:] # Eq.(28)  # division of corresponding column elements
    print('\n\ncii=', cii, '\nc_SPE\n', c_SPE, '\nRBC_SPE\n', RBC_SPE)

    RBC_T2 = (np.dot(X, D)**2) / ((np.diag(D))[None,:]) # Eq.(29)
    print('RBC_T2=\n', RBC_T2)

    RBC_phi = (np.dot(X, PHI)**2) / ((np.diag(PHI))[None,:]) # Eq.(30)
    print('RBC_phi=\n', RBC_phi)


####
# 5. Simulation study from [1]
####
def simulateX_pub1():
    numsim = 1000
    numsimfault = 2000
    faultmagnitude = 5.0
    #np.random.seed(0123)
    M = np.array([  # lines=sensors, columns=intrincsic states
        [-0.2310, -0.0816, -0.2662],
        [-0.3241,  0.7055, -0.2158],
        [-0.2170, -0.3056, -0.5207],
        [-0.4089, -0.3442, -0.4501],
        [-0.6408,  0.3102,  0.2372],
        [-0.4655, -0.4330,  0.5938]])
    numsensors, numstates = M.shape

    # N o r m a l
    T = np.random.normal(loc=[0,0,0], scale=[1, 0.8, 0.6], size=(numsim,numstates))
    noise = np.random.normal(scale=0.2, size = (numsim,numsensors))
    signal = np.dot(T, M.T) + noise
    normalmean = signal.mean(axis=0)
    mormalstd = signal.std(axis=0, ddof=ddof_std)
    signal = (signal - normalmean) / mormalstd    # Standardize to zero mean, unit variance
    #print('Simulated signal: Mean=\n', np.mean(signal,axis=0), '\nStd=\n', np.std(signal,axis=0))
    normal = signal
    normal_labels = np.zeros([numsim, numsensors], dtype=int)

    # F a u l t s
    T = np.random.normal(loc=[0,0,0], scale=[1, 0.8, 0.6], size=(numsimfault,numstates))
    noise = np.random.normal(scale=1, size = (numsimfault,numsensors))
    signal = np.dot(T, M.T) + noise
    signal = (signal - normalmean) / mormalstd
    #Xi = np.random.choice(numsensors, size=numsimfault)
    fault_labels = np.zeros([numsimfault, numsensors], dtype=int)
    for i in range(numsimfault):
        f = np.random.random()*faultmagnitude
        xi = np.random.choice(numsensors)
        fault_labels[i][xi] = 1
        signal[i][xi] += f
    fault = signal
    return normal, normal_labels, fault, fault_labels

####
# 5. Simulation study from [8]
####
def simulateX_pub8():
    numsimnormal = 1000
    #np.random.seed(0123)
    M = np.array([  # lines=sensors, columns=intrincsic states
        [-0.1670, -0.1352],
        [-0.5671, -0.3695],
        [-0.1608, -0.1019],
        [ 0.7574, -0.0563],
        [-0.2258,  0.9119]])
    numsensors, numstates = M.shape

    T = np.random.normal(loc=[0,0], scale=[1, 1], size=(numsimnormal,numstates))
    noise_variance = 0.1
    noise = np.random.normal(scale=np.sqrt(noise_variance), size=(numsimnormal,numsensors))
    signal = np.dot(T, M.T) + noise
    normal = signal
    fault = None
    return normal, fault

####
# 5. Simulation study from [9]
####
def simulateX_pub9():
    numsim = 100
    #np.random.seed(0123)
    M = np.array([  # lines=sensors, columns=intrincsic states
        [ 0.3873,  0.1190],
        [-0.1291,  0.2379],
        [ 0.9037, -0.1530],
        [ 0.1291,  0.9518]])
    numsensors, numstates = M.shape

    T = np.random.normal(loc=[0,0], scale=[1, 0.8], size=(numsim,numstates))
    noise = np.random.normal(scale=0.2, size = (numsim, numsensors))
    signal = np.dot(T, M.T) + noise
    #signal = (signal - signal.mean(axis=0)) / signal.std(axis=0, ddof=ddof_std)    # Standardize to zero mean, unit variance
    print('Simulated signal: Mean=\n', np.mean(signal,axis=0), '\nStd=\n', np.std(signal,axis=0, ddof=ddof_std))
    normal = signal
    fault = None
    return normal, fault




# Calculate the index of a whole data set X with 
def Index(X, M, fault_detect_only=False):    # Eq. (9)
    # Each line is equivalent calculus x'Mx as in a loop for each x
    Index_per_variable = np.dot(X, M) * X
    Index = np.sum( Index_per_variable, axis=1) # Eqs. 4,8,9,10,12
    if fault_detect_only:
        return Index
    #return np.sum( np.dot(X, M) * X, axis=1) # Eqs. 4,8,9,10,12
    return Index, Index_per_variable

def detected_faults( index, control_limit ):
    normal = index < control_limit
    num_faults = list(normal).count(False)
    fault_ratio = 1. * num_faults / index.shape[0]
    return normal, num_faults, fault_ratio



def fault_detect(training):
    #---------------
    # SPE
    #---------------
    M_c_SPE = training.get('M_c_SPE')
    deltasqr = training.get('deltasqr')
    training_data = training.get('Xtrain_norm')
    fault_data = training.get('Xtest_norm')

    SPE_c_X_training, SPE_c_i_X_training = Index( training_data, M_c_SPE)
    #print('SPE_c_X_training(', SPE_c_X_training.shape, ')=\n', SPE_c_X_training,
    #        '\nSPE_c_i_X_training(', SPE_c_i_X_training.shape, ')=\n', SPE_c_i_X_training)

    normal, num_faults, fault_ratio = detected_faults( SPE_c_X_training, deltasqr )
    print('# Training Faults for SPE / numNormalSPE=',
        num_faults, '/', training_data.shape[0], '=', 100*fault_ratio,'%')


    SPE_c_X_test = Index( fault_data, M_c_SPE, fault_detect_only=True )
    #SPE_c_X_test, SPE_c_i_X_test = Index( fault_data, M_c_SPE)
    #print('SPE_c_X_test(', SPE_c_X_test.shape, ')=\n', SPE_c_X_test,
    #        '\nSPE_c_i_X_test(', SPE_c_i_X_test.shape, ')=\n', SPE_c_i_X_test)

    normal, num_faults, fault_ratio = detected_faults( SPE_c_X_test, deltasqr )

    print('# Test Faults for SPE / numTestSPE=',
        num_faults, '/', fault_data.shape[0], '=', 100*fault_ratio,'%')
    #---------------
    # T2
    #---------------
    M_c_T2 = training.get('M_c_T2')
    tau2 = training.get('tau2')

    T2_c_X_training = Index( training_data, M_c_T2, fault_detect_only=True )
    normal, num_faults, fault_ratio = detected_faults( T2_c_X_training, tau2 )
    print('# Training Faults for T2 / numTrainingT2=',
            num_faults, '/', training_data.shape[0], '=', 100*fault_ratio,'%')

    T2_c_X_test = Index( fault_data, M_c_T2, fault_detect_only=True )
    #T2_c_X_test, T2_c_i_X_test = Index( fault_data, M_c_T2)

    normal, num_faults, fault_ratio = detected_faults( T2_c_X_test, tau2 )

    print('# Test Faults for T2 / numTestT2=',
        num_faults, '/', fault_data.shape[0], '=', 100*fault_ratio,'%')


    #---------------
    # Combined
    #---------------
    M_c_combined = training.get('M_c_combined')
    zeta2 = training.get('zeta2')

    combined_c_X_training = Index( training_data, M_c_combined, fault_detect_only=True )
    normal, num_faults, fault_ratio = detected_faults( combined_c_X_training, zeta2 )
    print('# Training Faults for Combined / numTrainingCombined=',
            num_faults, '/', training_data.shape[0], '=', 100*fault_ratio,'%')

    combined_c_X_test = Index( fault_data, M_c_combined, fault_detect_only=True )
    #combined_c_X_test, combined_c_i_X_test = Index( fault_data, M_c_combined)

    normal, num_faults, fault_ratio = detected_faults( T2_c_X_test, zeta2 )

    print('# Test Faults for Combined / numTestCombined=',
        num_faults, '/', fault_data.shape[0], '=', 100*fault_ratio,'%')


    #---------------
    # M2
    #---------------
    M_c_M2 = training.get('M_c_M2')
    M2_control_limit = training.get('M2_control_limit')
    M2_X_training = Index( training_data, M_c_M2, fault_detect_only=True )
    normal, num_faults, fault_ratio = detected_faults( M2_X_training, M2_control_limit )
    print('# Training Faults for M2 / numTrainingM2=',
            num_faults, '/', training_data.shape[0], '=', 100*fault_ratio,'%')

    M2_X_test = Index( fault_data, M_c_M2, fault_detect_only=True )
    normal, num_faults, fault_ratio = detected_faults( M2_X_test, M2_control_limit )
    print('# Test Faults for M2 / numTestM2=',
            num_faults, '/', fault_data.shape[0], '=', 100*fault_ratio,'%')





def main():
    print('Executing main() ....')

    #test(); quit()


    # Generate training and test data by simulation
    training_data, training_labels, fault_data, fault_labels = simulateX_pub1()
    #print('training_data=\n', training_data, '\ntraining_labels=\n', training_labels,
    #        '\nfault_data=\n', fault_data, '\nfault_labels=\n', fault_labels)

    print('\n\nFAULT DETECTION\n')
    # Fault detection
    training = gen_train(training_data, fault_data, num_principal_components=2) #; quit()
    #training = train_reconstruction_based(training) # calculate also the reconstruction-based info

    plot_2_D_statistics(training) #; quit()

    fault_detect(training)

    detect_contrib(training)




if __name__ == "__main__":
    main()




# S P E  C O N T R I B U T I O N - B A S E D




#      
#      
#T2_X = training.get('T2_X')
#tau2 = training.get('tau2')
#NormalConditionT2 = T2_X < tau2
#numFaultsT2 = list(NormalConditionT2).count(False)
##print('Normal Condition: T2 < tau2=', NormalConditionT2)
#print('# Faults for T2 / numNormalT2=', numFaultsT2, '/', T2_X.shape[0])
#
#combined_phi_X = training.get('combined_phi_X')
#zeta2 = training.get('zeta2')
#NormalConditionCombined = combined_phi_X < zeta2
#numFaultsCombined = list(NormalConditionCombined).count(False)
##print('Normal Condition: combined_phi_X < zeta2=', NormalConditionCombined)
#print('# Faults for combined index / numNormalCombined=', numFaultsCombined, '/', combined_phi_X.shape[0])
