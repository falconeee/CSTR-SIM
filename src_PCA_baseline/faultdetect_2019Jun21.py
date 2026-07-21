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
    
[2] Journal of Process Control, Volume 20, Issue 10, December 2010, Pages 1198-1206
    A Branch and Bound Method for Isolation of Faulty Variables through Missing Variable Analysis
    Vinay Kariwala , Pabara-Ebiere Odiowei , Yi Cao and Tao Chen

[3] Reconstruction-Based Fault Identification Using a Combined Index,
    H. Henry Yue, S. Joe Qin
    Ind. Eng. Chem. Res. 2001, 40, 4403-4414

[4] Statistical process monitoring: basics and beyond
    S. Joe Qin,
    J. Chemometrics 2003; 17: 480–502 , DOI: 10.1002/cem.800

[5] A comparison study of basic data-driven fault diagnosis and process monitoring
    methods on the benchmark Tennessee Eastman process
    Shen Yin, Steven X. Ding, Adel Haghani, Haiyang Hao, Ping Zhang
    Journal of Process Control 22 (2012) 1567–1581

========================================

"""

import sys
from datetime import datetime
import time
import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.stats import chi2
#import sklearn.datasets as datasets
from sklearn.decomposition import PCA
from socket import gethostname

datetime = datetime.now().strftime('%Y_%m_%d__%H:%M:%S')

hostname = gethostname()
if hostname == 'eos' or hostname == 'phoenix':
    logrootdir = '/home/thomas/ninfabox/experimental_results/'
else:
    logrootdir = '/export/thomas/experimental_results/'
dropfigdir = logrootdir


outfile = None
outfile = sys.stdout

if outfile != sys.stdout:
    logfname = logrootdir+datetime
    logfname = logfname+'.log'
    print('=== Opening log file', logfname)
    logfile = open(logfname, 'w')
    outfile = logfile

#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# If 0 < n_components < 1 and svd_solver == 'full',
# select the number of components such that the amount of variance that
# needs to be explained is greater than the percentage specified by n_components.

#PPCA_svd_solver = 'full'
#PPCA_n_components = 'mle'
#PPCA_n_components = 0.99
#PPCA_n_components = 7


no_graph = False
faultarg = None # can pass the fault number as parameter for TE

# LaTeX support: https://matplotlib.org/users/usetex.html
usetex = True
#usetex = False

useLaTex_for_documentation = True
useLaTex_for_documentation = False

def fprintf(*objects, sep=' ', end='\n', file=outfile, flush=False):
    print(*objects, sep=sep, end=end, file=file, flush=flush)

def print_array(x, formatstr='%.2f'):
    fprintf(np.array2string(x, formatter={'float_kind':lambda x: formatstr % x}))

def feats(featset):
    return tuple(1+featset.astype(int))

def cm2inch(cm):
    return cm/2.54

def tex_setup(usetex=True):
    if not usetex:
        fprintf('Not setting any Matplotlib parameters for LaTeX ...')
        return
    #plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    ## for Palatino and other serif fonts use:
    #plt.rc('font',**{'family':'serif','serif':['Palatino']})
    plt.rc('font',**{'family':'serif','serif':['DejaVu Sans']})
    plt.rc('text', usetex=usetex)
    # Set the font size. Either an relative value of 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large' or an absolute font size, e.g., 12.
    # https://matplotlib.org/api/font_manager_api.html#matplotlib.font_manager.FontProperties.set_size
    texfigparams = {'legend.fontsize': 6,
         'legend.loc': 'upper left',
         'figure.figsize': (8, 8),
         'axes.labelsize': 8,
         'axes.titlesize': 8,
         'lines.linewidth': 0.5,
         'xtick.labelsize': 7,
         'ytick.labelsize': 7}
    plt.rcParams.update(texfigparams)
    plt.rc('text', usetex=usetex)
    plt.rc('font', family='sans-serif')
    #print('Matplotlib: rcParams=\n', plt.rcParams, file=sys.stderr)


#ddof_std = 1        #  ==> divide by (n-1) --- ddof=0 ==> divide by n
ddof_std = 0       # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.std.html#numpy.std

def is_pos_definite(M):  # Tests if matrix is positive definite
    return np.all(np.linalg.eigvals(M) > 0)

# Python numpy.linalg.eig does not sort the eigenvalues and eigenvectors
def eigen(A):
    eigenValues, eigenVectors = LA.eig(A)
    idx = np.argsort(eigenValues)
    idx = idx[::-1] # Invert from ascending to descending
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)

def gen_train(Xtrain, Xtest, n_components, svd_solver='full', fault_num=None, featname=None, verbose=True):
    fprintf('gen_train> Calculating all parameters for fault detection ...')

    training = {} # dictionary

    # Standardize training data to zero mean and unit variance in each component
    X = Xtrain
    m, n = X.shape # m samples, n variables, data matrix has dimension (m,n)
    meanX = np.mean(X, axis=0)
    stdX = X.std(axis=0, ddof=ddof_std)
    minX = X.min(axis=0)
    maxX = X.max(axis=0)
    Xtrain_centered = X - meanX
    Xtrain_norm = Xtrain_centered / stdX
    Xtest_centered = Xtest - meanX
    Xtest_norm = (Xtest - meanX) / stdX
    if verbose:
        print ('gen_train> Number of samples = ', m, 'Number of variables = ', n)
        fprintf('Dataset statistic:\n Mean=', meanX, '\nStandard deviation=\n', stdX)
        fprintf('Dataset statistic:\nMin=', minX, '\nMax=', maxX )
        fprintf('Dataset Xtrain=\n', X, 'shape=', X.shape, '\nDataset centralized Xtrain_centered=\n', Xtrain_centered)
        fprintf('Training dataset standardized Xtrain_norm=\n', Xtrain_norm)
        fprintf('Test Dataset Xtest=\n', Xtest, 'shape=', Xtest.shape, '\nDataset centralized Xtest_centered=\n', Xtest_centered)
        fprintf('Test dataset standardized Xtest_norm=\n', Xtest_norm)
        #quit()

    '''
    # If number of PCA not specified, use the 
    if n_components == None:

        use_acumulated_variance = True

        if use_acumulated_variance:
            n_components = 0.99 # if 0<n_com<1, consider the aculumated variance
            pca = PCA(n_components=n_components)
            pca.fit(Xtrain_norm)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            numPCminvar = 1 + np.argmax(cumvar>n_components)
            fprintf('Number of PCs with >= %.2f%%' % (100*n_components), 'variance = ', numPCminvar,
                'of a total of', cumvar.shape[0], 'components')

        else:
            # Alternative> Use the algorithm of Minka
            pca = PCA(n_components='mle', svd_solver='full')
            pca.fit(Xtrain_norm)
            fprintf('Number of PCs determined by the method of Minka')

        l = pca.n_components_
        from probabilistic_PCA import show_result
        show_result(pca)
    else:
        l = n_components
    '''

    # [10] M2 statistic (Squared Mahalanobis Distance)
    from probabilistic_PCA import PPCA_parameters, PPCA
    n_samples, d, M2_control_limit = PPCA_parameters(Xtrain)
    mu, W, sigma2noise, C, Cinv, numeigval = PPCA(Xtrain, n_components=n_components, svd_solver=svd_solver, outfile=outfile)
    training.update({'M2_control_limit': M2_control_limit})
    M_c_M2 = Cinv
    if benchmark == 'pubBnB':   # Special case: use the explicit C matrix [1], E. (29)
        C_from_pub = get_pubBnB_C_matrix()
        M_c_M2 = LA.inv(C_from_pub)
        M_c_M2 = Cinv
        fprintf('M2: C from data=\n', C, '\nDifference from pub=\n', C-C_from_pub)
    training.update({'M_c_M2': M_c_M2})

    l = numeigval

    '''
    if isinstance(n_components, int):
        l = n_components
    else:
        pca = PCA(n_components=n_components, svd_solver=svd_solver)
        pca.fit(Xtrain)
        l = numeigval = pca.n_components_
    '''

    print('gen_train> n_components=', n_components, 'l=', l) # ; quit()

    training.update({'fault_num': fault_num})
    training.update({'featname': featname})
    training.update({'num_samples': m})
    training.update({'num_variables': n})
    training.update({'n_components': n_components})
    training.update({'estimated_num_principal_components': l})
    training.update({'Xtrain': Xtrain})
    training.update({'Xtest': Xtest})
    training.update({'meanX': meanX})
    training.update({'stdX': stdX})
    training.update({'minX': minX})
    training.update({'maxX': maxX})
    training.update({'Xtrain_centered': Xtrain_centered})
    training.update({'Xtest_centered': Xtest_centered})
    training.update({'Xtrain_norm': Xtrain_norm})
    training.update({'Xtest_norm': Xtest_norm})


    # 2.2.1. Eq. (4) Squared prediction error, SPE
    S = np.cov(Xtrain_norm, rowvar=False)
    Lambda, Phi = eigen(S)
    training.update({'Lambda': Lambda})
    training.update({'Phi': Phi})

    P = Phi[:, :l]
    L = np.diag(Lambda[:l])
    if verbose:
        fprintf('Lambda=\n', Lambda, 'shape=', Lambda.shape, '\nPhi=\n', Phi, 'shape=', Phi.shape)
        fprintf('First ', l, 'columns of Loading matrix Phat=First  ', l, 'eigenvectors=\n', P)
        fprintf('Diagonal matrix of first ', l, 'eigenvalues L=\n', L)
    
    # T2 statistic
    LInv = np.diag((1.0/Lambda[:l]))
    D = np.dot(np.dot(P, LInv), P.T)
    M_c_T2 = D 

    
    Ptil = Phi[:, l:]
    Ltil = np.diag(Lambda[l:])
    training.update({'P': P})
    training.update({'Ptil': Ptil})
    training.update({'L': L})
    training.update({'Ltil': Ltil})
    
    Shat = np.dot(np.dot(P, L), P.T)
    Stil = np.dot(np.dot(Ptil, Ltil), Ptil.T)
    #fprintf('Verify (should be zero): S=Shat+Stil\n', (Shat+Stil)-S) # Eq. (3)
    
    C = np.dot(P, P.T)  # C is identity if all components are principal since PHI*PHI'=I
    Ctil = np.dot(Ptil, Ptil.T)
    M_c_SPE = Ctil
    if verbose:
        fprintf('Diagonal matrix of the inverse of the first ', l, 'eigenvalues L=\n', LInv)
        fprintf('Matrix D = PL^-1P''=\n', D)
        fprintf('T2 whitening subspace projection matrix M_T2=\n', M_c_T2)
        fprintf('Last ', n-l, 'columns of Loading matrix Phat=\n', Ptil)
        fprintf('Diagonal matrix of last ', n-l, 'eigenvalues Ltil=\n', Ltil)
        fprintf('Shat=', Shat)
        fprintf('Stil=', Stil)
        fprintf('Principal component subspace (PCS) projection matrix C=\n', C)
        fprintf('Residual subspace (RS) projection matrix M_c_SPE=Ctil=\n', Ctil)
        fprintf('Verify Ctil*Ctil''= Ctil: norm of diff matrix=', LA.norm(np.dot(Ctil, Ctil)-Ctil))
    
    
    # 2.2.1. Eq. (4) Squared prediction error, SPE
    
    theta1 = np.sum(Lambda[l:])
    theta2 = np.sum(np.square(Lambda[l:]))
    g_SPE = theta2 / theta1
    h_SPE = theta1**2 / theta2
    alpha = 0.05
    deltasqr = g_SPE * chi2.ppf(1-alpha, h_SPE)
    if verbose:
        fprintf('theta1=', theta1, 'theta2=', theta2, 'g_SPE=', g_SPE,
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
    if verbose:
        fprintf('Braatz et al.: l=a=', l, 'm=n=', m, 'tau2 Braatz=', tau2)
    
    tau2 = chi2.ppf(1-alpha, l) # Qin
    if verbose:
        fprintf('tau2 Qin=', tau2)
    
    training.update({'tau2': tau2})
    training.update({'M_c_T2': M_c_T2})
    
    # Combined index: [1] Eq. (6)
    PHI = Ctil/deltasqr + D/tau2
    M_c_combined = sqrtm(PHI)
    if verbose:
        fprintf('Combined index matrix: PHI=\n', PHI)
        fprintf('Combined projection matrix M_c_combined=\n', M_c_combined)
    
    # Combined index control limit
    aux1 = 1.0/tau2**2 + theta2/deltasqr**2
    aux2 = 1.0/tau2 + theta1/deltasqr
    g_phi = aux1 / aux2
    h_phi = aux2**2 / aux1
    zeta2 = g_phi * chi2.ppf(1-alpha, h_phi)
    if verbose:
        fprintf('g_phi=', g_phi, 'h_phi=', h_phi, 'zeta2=', zeta2)
    
    training.update({'zeta2': zeta2})
    training.update({'M_c_combined': M_c_combined})

    fprintf('\nT R A I N I N G  P A R A M E T E R S  L E A R N E D\n')
    return training


def Index_and_individual_contribs(X, Msqrt, verbose=False):    # [1] Eq. (8),(9),(10)
    #fprintf('Msqrt=\n', Msqrt, '\nMsqrt^2=', np.dot(Msqrt,Msqrt))
    #fprintf('X.shape=', X.shape, 'X=\n', X)

    individual_contribs = np.dot(X, Msqrt)**2
    if verbose:
        fprintf('\nIndividual_contribs=', individual_contribs, 'shape=', individual_contribs.shape)
        #fprintf('Msqrt=\n', Msqrt, '\nX[0]=\n', X[0], '\nindividual_contribs[0]=', individual_contribs[0])
    Index = np.sum(individual_contribs, axis=1)
    #M=np.dot(Msqrt,Msqrt); Index2 = np.sum(np.dot(X,M)*X, axis=1); diff=LA.norm(Index-Index2);fprintf('@@@diff=', diff); # works
    if verbose:
        fprintf('Index=', Index, 'shape=', Index.shape)
        #fprintf('Index[0]=', Index[0])
    return Index, individual_contribs


def detect_contrib(training):
    #  [1]  2.3. Fault diagnosis by contribution plots   
    fprintf('\n\n --- [1]  2.3. Fault diagnosis by contribution plots ---\n\n')
    
    Ctil = M_c_SPE = training.get('M_c_SPE')
    X = training.get('Xtrain_norm')

    # SPE contribution
    #fprintf('Ctil=M_c_SPE=\n', Ctil)
    Xtil = np.dot(X, Ctil) # Eq. (4)
    c_SPE = Xtil**2 # Eq. (10), (12) element-wise square, each line
    #                   contains [c_0, c_1, ...] of all the samples x
    SPE = np.sum(c_SPE, axis=1) # Eq. (4) SPE statistic of each sample
    #fprintf('Xtil=\n', Xtil, '\nindividual contribs c_SPE=\n', c_SPE, '\nSPE=', SPE)
    fprintf('detect_contrib> c_SPE[0]=', c_SPE[0], 'SPE[0]=', SPE[0], 'limit=',training.get('deltasqr'))

    SPE_new, c_SPE_new = Index_and_individual_contribs(X, Ctil)
    #diff = c_SPE - c_SPE_new
    #fprintf('SPE diff=', c_SPE - c_SPE_new, '\n norm=', LA.norm(diff),
    #        '\n', SPE-SPE_new, '\nnorm=', LA.norm(SPE-SPE_new)); quit()
    
    # T2 contribution
    D = training.get('M_c_T2')
    Dsqrt = sqrtm(D) # cholesky does not work
    c_T2 = np.dot(X, Dsqrt)**2  # Eq. (5)
    T2 = np.sum(c_T2, axis=1)
    #fprintf('c_T2.shape=\n', c_T2.shape, 'individual contribs c_T2=\n', c_T2)
    fprintf('detect_contrib> training data: c_T2[0]=', c_T2[0], 'T2[0]=', T2[0], 'limit=', training.get('tau2'))

    '''
    T2new, c_T2_new = Index_and_individual_contribs(X, D)    # [1] Eq. (13),(5)
    fprintf('diff=', c_T2 - c_T2new, '\n\nT2=', T2_new, 'T2_new.shape=', T2_new.shape)
    quit()
    '''
    
    # Combined index contribution
    PHI = training.get('M_c_combined')
    c_phi = np.dot(X, sqrtm(PHI))**2
    phi = np.sum(c_phi, axis=1) # Eq. (6)
    '''
    fprintf('c_phi=\n', c_phi, '\nphi=', phi)
    phi_new, c_phi_new = Index_and_individual_contribs(X, PHI)
    fprintf('diff phi norm=', LA.norm(c_phi-c_phi_new), LA.norm(phi-phi_new))
    quit()
    '''

    # M2 contribution   [2], Eq. 3
    M_c_M2 = training.get('M_c_M2')
    X = training.get('Xtrain') - training.get('meanX')
    #fprintf('M_c_M2=', M_c_M2, '\nX=', X);# quit()
    M2 = np.sum(np.dot(X, M_c_M2) * X, axis=1)


####
# Simulation study from [1], p. 1598
# Modification: In the faults, only a subset of fault directions are allowed,
# not any randomly chosen fault direction as in the original simulation
####
def simulateX_pub1(faultdirections='Random'):
    numsim = 1000
    numsimfault = 2000
    #numsimfault = 20
    faultmagnitude = 5.0
    np.random.seed(seed=66649)
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
    #fprintf('Simulated signal: Mean=\n', np.mean(signal,axis=0), '\nStd=\n', np.std(signal,axis=0))
    normal = np.copy(signal)
    normal_labels = np.zeros([numsim, numsensors], dtype=int)

    # F a u l t s
    if faultdirections != 'Random':
        num_fault_directions = len(faultdirections)

    T = np.random.normal(loc=[0,0,0], scale=[1, 0.8, 0.6], size=(numsimfault,numstates))
    noise = np.random.normal(scale=1, size = (numsimfault,numsensors))
    signal = np.dot(T, M.T) + noise
    signal = (signal - normalmean) / mormalstd
    #Xi = np.random.choice(numsensors, size=numsimfault)
    fault_labels = np.zeros([numsimfault, numsensors], dtype=int)
    for i in range(numsimfault):
        if faultdirections == 'Random':
            f = np.random.random()*faultmagnitude
            xi = np.random.choice(numsensors)
            #fprintf('pub_Qin_Automatica: random fault sample #%4d' % (i+1), 'mag=%7.3f' % f, 'direction=', xi+1)
            fault_labels[i][xi] = 1
            signal[i][xi] += f
        else:
            for j in range(num_fault_directions):
                direction = faultdirections[j]-1
                f = np.random.random()*faultmagnitude
                signal[i][direction] += f
                fault_labels[i][direction] = 1
        #fprintf('pub_Qin_Automatica: directed fault sample #%4d' % (i+1), 'signal=\n', signal, '\nfault_labels=\n', fault_labels)

    fault = np.copy(signal)
    return normal, normal_labels, fault, fault_labels

####
# Simulation study from [3] , and [2], p. 11
####
def get_pubBnB_fault_data(use_only_four_vars=False):
    #fprintf('get_pubBnB_fault_data> single fault value from pub with magnitude 1.8 in fourth sensor')
    #fault = np.array([[-0.079, -0.59, -0.22, -1.78, -0.024]]) 

    fprintf('get_pubBnB_fault_data>  multi-sensor fault with magnitude 1.5 in third and fourth sensor')
    fault = np.array([[-0.079, -0.59, 1.49, -1.48, -0.024]])

    #fprintf('get_pubBnB_fault_data> multi-sensor fault with magnitude 0.5 in third and fourth sensor')
    #fault = np.array([[-0.079, -0.59, 0.49, -0.48, -0.024]])
    if use_only_four_vars:
        fault = fault[0][:4].reshape(1,4)

    return fault

def get_pubBnB_C_matrix():
    fprintf('get_pubBnB_C_matrix> Setting C matrix as [1], Eq. (29)')
    C = np.array([[0.060400, 0.154800, 0.043500, -0.124700, -0.098300],
        [0.154800, 0.496300, 0.136900, -0.427000, -0.240000],
        [0.043500, 0.136900, 0.049100, -0.122500, -0.063400],
        [-0.124700, -0.427000, -0.122500, 0.599700, -0.202000],
        [-0.098300, -0.240000, -0.063400, -0.202000, 0.926200]])
    fprintf('\n\nFixed C from pub [1],Eq.(29): C=\n', C)
    return C

def simulateX_pubBnB(use_only_four_vars=False):
    numsimnormal = 1000
    np.random.seed(seed=66649)
    G = np.array([  # lines=sensors, columns=intrincsic states
        [-0.1670, -0.1352],
        [-0.5671, -0.3695],
        [-0.1608, -0.1019],
        [ 0.7574, -0.0563],
        [-0.2258,  0.9119]])
    if use_only_four_vars:
        G = G[:4,:]
    numsensors, numstates = G.shape

    # Attention: Difference for Matlab and Python:
    # scale is standard deviation, not variance
    variance = 1.0
    stdev = scale = np.sqrt(variance)

    T = np.random.normal(loc=[0,0], scale=[scale, scale], size=(numsimnormal,numstates))
    #fprintf('\nT.shape=', T.shape, 'T=\n', T)
    sensor_error_variance = 0.01
    sensor_error = np.random.normal(scale=np.sqrt(sensor_error_variance), size = (numsimnormal,numsensors))
    signal = np.dot(T, G.T) + sensor_error
    normal = signal
    fault = get_pubBnB_fault_data(use_only_four_vars)
    '''
    fprintf('\nsignal.shape=', signal.shape, 'signal=\n', signal)
    fprintf('\nuse_only_four_vars=', use_only_four_vars, 'fault.shape=', fault.shape, 'fault=\n', fault)
    #quit()
    '''
    normalmean = signal.mean(axis=0)
    normalstd = signal.std(axis=0, ddof=ddof_std)
    return normal, fault

####
# Simulation study from [4]
####
def simulateX_pub9():
    numsim = 100
    np.random.seed(seed=123)
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
    fprintf('Simulated signal: Mean=\n', np.mean(signal,axis=0), '\nStd=\n', np.std(signal,axis=0, ddof=ddof_std))
    normal = signal
    fault = None
    return normal, fault



def detected_faults( index, control_limit ):
    normal = index < control_limit
    num_faults = list(normal).count(False)
    fault_ratio = 1. * num_faults / index.shape[0]
    return normal, num_faults, fault_ratio


def get_mat_and_limit(training, fault_detection_index='SPE'):
    if fault_detection_index == 'SPE':
        M = training.get('M_c_SPE')
        control_limit = training.get('deltasqr')
        M_sqrt = M
        
    elif fault_detection_index == 'T2':
        M = training.get('M_c_T2')
        control_limit = training.get('tau2')
        M_sqrt = sqrtm(M)
        
    elif fault_detection_index == 'Combined':
        M = training.get('M_c_combined')
        control_limit = training.get('zeta2')
        M_sqrt = sqrtm(M)
        
    elif fault_detection_index == 'M2':
        M = training.get('M_c_M2')
        control_limit = training.get('M2_control_limit')
        M_sqrt = sqrtm(M)
    else:
        fprintf('Contribution plot: Unknown Index'); return
    #fprintf('DEBUG: get_mat_and_limit> Test is matrix is positive definite')
    #fprintf('@@@fault_detection_index=', fault_detection_index, 'M: ', is_pos_definite(M), 'sqrt(M): ', is_pos_definite(M_sqrt))
    return M_sqrt, control_limit, M




def calc_all_fault_detection_indices(training):
    
    training_data = training.get('Xtrain_norm')
    fault_data = training.get('Xtest_norm')

    #---------------
    # SPE
    #---------------
    M_sqrt, control_limit, M = get_mat_and_limit(training, fault_detection_index='SPE')

    SPE_c_X_training, SPE_c_i_X_training = Index_and_individual_contribs( training_data, M_sqrt)
    '''
    fprintf('SPE_c_X_training(', SPE_c_X_training.shape, ')=\n', SPE_c_X_training)
    fprintf('\nSPE_c_i_X_training(', SPE_c_i_X_training.shape, ')=\n', SPE_c_i_X_training)
    fprintf('fault_detect> training data: c_SPE[0]=', SPE_c_i_X_training[0], 'SPE[0]=', SPE_c_X_training[0], 'limit=', deltasqr)
    '''
    
    normal, num_faults, fault_ratio = detected_faults( SPE_c_X_training, control_limit )
    fprintf('# Training Faults for SPE / numNormalSPE=',
        num_faults, '/', training_data.shape[0], '=', 100*fault_ratio,'%')

    SPE_c_X_test, SPE_c_i_X_test  = Index_and_individual_contribs( fault_data, M_sqrt )

    #fprintf('SPE_c_X_test(', SPE_c_X_test.shape, ')=\n', SPE_c_X_test)
    #fprintf('\nSPE_c_i_X_test(', SPE_c_i_X_test.shape, ')=\n', SPE_c_i_X_test)

    normal, num_faults, fault_ratio = detected_faults( SPE_c_X_test, control_limit )

    fprintf('# Test Faults for SPE / numTestSPE=',
        num_faults, '/', fault_data.shape[0], '=', 100*fault_ratio,'%')

    #---------------
    # T2
    #---------------
    M_sqrt, control_limit, M = get_mat_and_limit(training, fault_detection_index='T2')

    T2_c_X_training, T2_c_i_X_training = Index_and_individual_contribs( training_data, M_sqrt )
    normal, num_faults, fault_ratio = detected_faults( T2_c_X_training, control_limit )
    #fprintf('fault_detect> training data: c_T2[0]=', T2_c_i_X_training[0], 'T2[0]=', T2_c_X_training[0], 'limit=', control_limit)
    fprintf('# Training Faults for T2 / numTrainingT2=',
            num_faults, '/', training_data.shape[0], '=', 100*fault_ratio,'%')

    T2_c_X_test, T2_c_i_X_test = Index_and_individual_contribs( fault_data, M_sqrt )

    normal, num_faults, fault_ratio = detected_faults( T2_c_X_test, control_limit )

    fprintf('# Test Faults for T2 / numTestT2=',
        num_faults, '/', fault_data.shape[0], '=', 100*fault_ratio,'%')

    #---------------
    # Combined
    #---------------
    M_sqrt, control_limit, M = get_mat_and_limit(training, fault_detection_index='Combined')

    combined_c_X_training, combined_c_i_X_training = Index_and_individual_contribs( training_data, M_sqrt )
    normal, num_faults, fault_ratio = detected_faults( combined_c_X_training, control_limit )
    fprintf('# Training Faults for Combined / numTrainingCombined=',
            num_faults, '/', training_data.shape[0], '=', 100*fault_ratio,'%')

    combined_c_X_test, combined_c_i_X_test = Index_and_individual_contribs( fault_data, M_sqrt )

    normal, num_faults, fault_ratio = detected_faults( T2_c_X_test, control_limit )

    fprintf('# Test Faults for Combined / numTestCombined=',
        num_faults, '/', fault_data.shape[0], '=', 100*fault_ratio,'%')

    #---------------
    # M2
    #---------------
    #M_sqrt, control_limit, M = get_mat_and_limit(training, fault_detection_index='M2')
    M_c_M2 = training.get('M_c_M2')
    M2_control_limit = training.get('M2_control_limit')

    meanX = training.get('meanX')
    training_data = training.get('Xtrain') - meanX
    fault_data = training.get('Xtest') - meanX

    X = training_data
    #fprintf('X.shape=', X.shape, 'M_c_M2.shape=', M_c_M2.shape)
    M2_X_training =  np.sum( np.dot(X, M_c_M2) * X, axis=1 )
    #fprintf('M2_X_training=\n', M2_X_training, 'shape=', M2_X_training.shape); quit()

    normal, num_faults, fault_ratio = detected_faults( M2_X_training, M2_control_limit )
    fprintf('# Training Faults for M2 / numTrainingM2=',
            num_faults, '/', training_data.shape[0], '=', 100*fault_ratio,'%')

    X = fault_data
    M2_X_test =  np.sum( np.dot(X, M_c_M2) * X, axis=1 )
    normal, num_faults, fault_ratio = detected_faults( M2_X_test, M2_control_limit )
    fprintf('# Test Faults for M2 / numTestM2=',
            num_faults, '/', fault_data.shape[0], '=', 100*fault_ratio,'%')


def contribution_and_time_plot(training, fault_data, discard_first_test=None, fault_start=None, window=1,
        fault_detection_index='SPE',
        semilogy=False, benchmark=None, dropfigdir=None, figext='.svg'):

    fault_stop = fault_start+window

    if fault_detection_index == 'M2':
        meanX = training.get('meanX')
        #fprintf('M2 index: Special non-standardized training and test data...')
        training_data = training.get('Xtrain') - meanX

        '''
        #np.set_printoptions(threshold=sys.maxsize)
        print('Before M2 centralize: fault_data=\n', fault_data, 'shape=', fault_data.shape, '\nmeanX=\n', meanX)
        x = fault_data[0]; print('fault_data[0]='); print_array(x, formatstr='%.2f')
        x = fault_data[160]; print('fault_data[160]='); print_array(x, formatstr='%.2f')
        '''

        fault_data = fault_data - meanX

        '''
        print('After M2 centralize: fault_data=\n', fault_data, 'shape=', fault_data.shape)
        Cinv = training.get('M_c_M2')
        print('Cinv=\n', Cinv, 'shape=', Cinv.shape)
        x = fault_data[0]; print('fault_data[0]='); print_array(x, formatstr='%.2f')
        M2 = np.dot(np.dot(x, Cinv), x.T); print('\nM2=', M2)
        x = fault_data[160]; print('fault_data[160]='); print_array(x, formatstr='%.2f')
        M2 = np.dot(np.dot(x, Cinv), x.T); print('\nM2=', M2)
        '''
    else:
        training_data = training.get('Xtrain_norm')

        #print('Before standardize: fault_data=\n', fault_data, 'shape=', fault_data.shape)
        fault_data = standardize(training, fault_data, fault_detection_index)
        #print('After standardize: fault_data=\n', fault_data, 'shape=', fault_data.shape)

    if benchmark == 'pubBnB':   # Special case: use the explicit fault pattern
        fault_data_window = fault_data = get_pubBnB_fault_data()
    else:
        fault_data_window = fault_data[fault_start:fault_stop]

    #print('fault_data_window from ', fault_start, 'to', fault_stop, '=\n', fault_data_window)   ; quit()
    

    M_sqrt, control_limit, M = get_mat_and_limit(training, fault_detection_index)

    Index_train, Index_per_variable_train = Index_and_individual_contribs(training_data, M_sqrt)
    #print('Index_train=', Index_train, 'Index_train.shape=', Index_train.shape);
    Index_train = abs(Index_train)  # might get some complex values
    Index_per_variable_train = abs(Index_per_variable_train)

    Index_test, Index_per_variable_test = Index_and_individual_contribs(fault_data, M_sqrt, verbose=False)
    Index_test = abs(Index_test)  # might get some complex values
    Index_per_variable_test = abs(Index_per_variable_test)
    #print('Contribution plot> Index_test=\n', Index_test, 'shape=', Index_test.shape) #; quit()


    # C o n t r i b u t i o n  p l o t 
    #cplot, ax = plt.subplots(); # Create a figure and a set of subplots
    cplot, ax = plt.gcf(), plt.gca()    # Current figure and axes

    # explicitly choose the sample for which the individual contributions are to be plotted
    #fprintf('individual_contribs=', individual_contribs)

    # [2], Fig.2
    tex_setup(usetex=usetex)
    
    print('Contribution plot from individual index from ', fault_start,
            'to', fault_stop, 'of', fault_data_window.shape[0], 'samples')
    individual_contribs = Index_per_variable_test[fault_start:fault_stop].mean(axis=0)

    height = individual_contribs
    numvar = training.get('num_variables')
    y_pos = np.arange(numvar)
    plt.bar(y_pos, height, color='#DDD0FF', label=None)
    ax = plt.gca()
    for i, v in enumerate(height):
        ax.text(i, height[i], '{:.2f}'.format(height[i]), fontsize=7, verticalalignment='bottom',
            horizontalalignment='center', alpha=1.0, rotation=0, color='black')

    plt.xticks(y_pos, 1+y_pos, rotation=90)

    barnames = training.get('featname')
    if not barnames is None:
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, ymax])   # Seems to be necessary to plot correctly
        yrange = ax.get_ylim()
        yoffset = 0.02*yrange[1]    # percentage of extension of y-axis
        plt.tick_params(labelbottom=False)#, bottom=False)
        #fprintf('yrange=', yrange, 'yoffset=', yoffset);
        for i, v in enumerate(height):
            ax.text(i, yoffset, barnames[i], fontsize=7, verticalalignment='bottom',
                    horizontalalignment='center', alpha=0.6, rotation=90, color='black')

    plt.xlabel('variable')
    plt.title('Individual Contribution Plot for Index ' + fault_detection_index)
    plt.ylabel('contribution')
    limit_label = fault_detection_index + ' control limit %.2f' % control_limit
    #if not control_limit is None:
    #    plt.axhline(y=control_limit, color='r', linestyle='--', label=limit_label)
    #    plt.legend()
    if not dropfigdir is None:
        # assemble file name of contribution plot with time stamp
        nowstr = datetime #.now().strftime('%Y_%m_%d__%H_%M_%S')
        fname = dropfigdir + nowstr + '_contrib' + figext
        #fname = '/home/thomas/Dropbox/papers/2019_tmp/figs/test1.eps'    # DEBUG
        fprintf('Saving contribution plot in ', fname)
        print('Saving contribution plot in ', fname)
        plt.savefig(fname, dpi=1200)
    plt.tight_layout()
    plt.show()


    # T i m e  E v o l u t i o n
    timeevolution, ax = plt.subplots(); # Create a figure and a set of subplots
    tex_setup(usetex=usetex)

    widthcm = 8 # Real sizes later in the LaTeX file
    heigthcm = 6
    timeevolution.set_size_inches([cm2inch(widthcm), cm2inch(heigthcm)])
    #timeevolution.tight_layout(pad=0, h_pad=0, w_pad=0)

    plt.title('Temporal Evolution of ' + fault_detection_index + ' Index')
    plt.xlabel('sample')
    plt.ylabel( fault_detection_index + ' Index')
    if control_limit != None:
        limit_label = 'Control limit'
        plt.axhline(y=control_limit, color='r', linestyle='--', label=limit_label, linewidth=0.75)
    numpoints_train = Index_train.shape[0]

    # End of training data
    plt.axvline(x=numpoints_train-1, color='cyan', linestyle='-', linewidth=0.5, label='End of training')
    if not discard_first_test is None:
        # Start of used test data for detection
        plt.axvline(x=numpoints_train-1+discard_first_test, color='magenta', linestyle='-', linewidth=0.5, label='Fault triggered')

    numpoints_test = Index_test.shape[0]
    xpostrain = np.linspace(0, numpoints_train-1, num=numpoints_train)
    xpostest = np.linspace(numpoints_train-1, numpoints_train+numpoints_test-1, num=numpoints_test)
    fprintf('Contribution plot: numpoints_train=', numpoints_train, 'numpoints_test=', numpoints_test)
    plotfunc = plt.plot
    if semilogy:
        plotfunc = plt.semilogy
    plotfunc(xpostrain, Index_train, linewidth=0.25, color='green', label='Normal')
    plotfunc(xpostest, Index_test, linewidth=0.25, color='blue', label='Test')

    # change the xticks of the fault data set
    xticks = ax.get_xticks().tolist()
    #print('xticks=', xticks)
    for i in range(len(xticks)):
        t = xticks[i]
        #print('t=', t)
        if t > numpoints_train:
            t -= numpoints_train
            #print('over new=', t)
        xticks[i] = int(t)
    ax.set_xticklabels(xticks)

    if not fault_stop is None:
        intvalcol = 'grey'
        intvalcol = '0.7'
        # Plot the interval limits of the test data used for the calculus of contribution
        #plt.axvline(x=numpoints_train-1+fault_start, color='cyan', linestyle='-', linewidth=0.75, label='Fault detection window start')
        #plt.axvline(x=numpoints_train-1+fault_stop, color='magenta', linestyle='-', linewidth=0.75, label='Fault detection window stop')

        startpos = numpoints_train-1+fault_start
        stoppos = numpoints_train+fault_stop
        intval = np.arange(startpos, stoppos)
        #ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, ymax])   # Seems to be necessary to plot correctly
        ax.fill_between(intval, ymin, ymax, color=intvalcol, alpha=0.3, label='Fault detection\nwindow')


    legend_loc = None
    legend_loc = 'best'
    legend_loc ='lower center'
    legend_loc ='upper left'
    #ax.legend(bbox_to_anchor=(0.0, 1.25))
    #plt.legend(loc=legend_loc, ncol=2)
    plt.legend(loc='upper right', bbox_to_anchor=(0.80, 1.05),  ncol=2)
    #plt.legend(loc='lower right', bbox_to_anchor=(0.80, 0.05),  ncol=2)
    if not dropfigdir is None:
        # assemble file name of time evolution plot with time stamp
        fname = dropfigdir + nowstr + '_evolution' + figext
        #fname = '/home/thomas/Dropbox/papers/2019_tmp/figs/test2.eps'    # DEBUG
        fprintf('Saving time evolution plot in ', fname)
        plt.savefig(fname, dpi=1200)
    plt.tight_layout()
    plt.show()


def detect_fault(training, fault_detection_index, fault_data, discard_first_test=None, window=1):
    '''Given a signal sequence and a window size, test if the mean
    of the signal in the window surpasses a threshold.
    Eventually discard the first parts of the fault signal.
    If a fault is detected return the value of the window mean and the position.
    The position is the absolute position of the fault signal
    '''

    #print('detect_fault> Before standardize: fault_data=\n', fault_data, 'shape=', fault_data.shape)

    # M2 is only centralized within the 'standardize' function !!!
    fault_data = standardize(training, fault_data, fault_detection_index)
    #print('detect_fault> After standardize: fault_data=\n', fault_data, 'shape=', fault_data.shape)

    #print('Before discard: Index verification test : M2 Index=\n', np.sum(np.dot(fault_data, training.get('M_c_M2')) * fault_data, axis=1) ) #; quit()

    # Test data 48h with 960 samples = 20 samples/h = 1 sample each 3 min --- Fault after 8h = 160 samples
    if not discard_first_test is None:
        len_before = len(fault_data)
        len_after = len_before - discard_first_test
        fprintf('Discarding the first', discard_first_test, 'samples from fault data. # samples before=', len_before, 'after=', len_after)
        fault_data = fault_data[discard_first_test:]

    M_sqrt, control_limit, M = get_mat_and_limit(training, fault_detection_index)
    #fprintf('detect_fault> control_limit=', control_limit)#, '\nM=\n', M, '\nM_sqrt=\n', M_sqrt)

    if benchmark == 'pubBnB':
        fault_data = get_pubBnB_fault_data()

    #fprintf('detect_fault> fault_data=\n', fault_data, 'shape=', fault_data.shape)
    Index, Index_per_variable = Index_and_individual_contribs(fault_data, M_sqrt, verbose=False)

    '''
    print('detect_fault> Index=', Index)
    print('detect_fault> Index=\n', Index, 'shape=', Index.shape)
    print('Individual contribution=\n', Index_per_variable, 'shape=', Index_per_variable.shape)
    print('Index verification test : Index=\n', np.sum(np.dot(fault_data,M) * fault_data, axis=1) )
    print('Index verification test : M2 Index=\n', np.sum(np.dot(fault_data, training.get('M_c_M2')) * fault_data, axis=1) ) #; quit()
    '''

    seqlen = fault_data.shape[0]
    assert seqlen >= window, str(seqlen)+' >= '+str(window)+' Window cannot be larger than sequence'

    if seqlen == 1:
        #m = seq.mean()  # For the case when there is only a single value
        x = fault_data[0]
        Index = np.dot(np.dot(x.T,M),x)
        fault = Index > control_limit
        #print('Index=', Index, 'fault=', fault)
        m = Index
        l = 0
        return fault, m, l

    n1 = 1. / window

    meanX = training.get('meanX')
    stdX = training.get('stdX')

    seq = np.zeros(seqlen)
    for i in range(window):
        x = fault_data[i]
        Index = np.dot(np.dot(x.T,M),x)
        seq[i] = Index
        #print('Index[', i, ']=', Index)

    sinit = seq[0:window-1]
    cumsum = np.cumsum(sinit)
    sinit = np.insert(cumsum, 0, [0])
    #print('Init: sinit=', sinit, '\nseq=', seq[:20]); quit()

    s = np.zeros(seqlen+1)
    s[0:window] = sinit
    l = 0
    r = window
    #print('\ns=', s[:20])
            
    done = False
    fault = False
    while not done:
        done = r == seqlen or fault
        if not done:
            x = fault_data[r]
            Index = np.dot(np.dot(x.T,M),x)
            seq[r-1] = Index

            s[r] = seq[r-1] + s[r-1]
            d = s[r] - s[l]
            m = d*n1
            fault = m >= control_limit
            #if fault:
            #    print('>>>>> fault=', fault, 'lim=', control_limit, 'l=', l, 'r=', r,
            #        'd=', d, 'mean Index m=', m, '\nseq=', seq[l:r]); #input('...')
            if not fault:
                r += 1
                l += 1
    if not discard_first_test is None:
        l += discard_first_test
    #print('Detection: fault=', fault, ', x dim=', len(x), 'mean index of', window, 'signals=', m, 'lim=', control_limit, 'left pos+1=', l+1, 'discard=', discard_first_test)
    return fault, m, l


def preselect(training, fault_data, fault_detection_index='SPE', maxpre=None, minmaxpercentage=0.15):
    '''Preselected are those variables with the HIGHEST individual contribution
    '''
    fprintf('--------------------------------------------------')
    fprintf('--- V A R I A B L E  P R E - S E L E C T I O N ---')
    fprintf('--------------------------------------------------')
    starttime = time.time()

    d = training.get('num_variables')
    contributing_preselection = np.ones(d, dtype=bool)
    #print('contributing_preselection=', contributing_preselection)

    M_sqrt, control_limit, M = get_mat_and_limit(training, fault_detection_index)

    # for M2 the data is only centralized !!!
    fault_data = standardize(training, fault_data, fault_detection_index)

    Index, contrib = Index_and_individual_contribs(fault_data, M_sqrt, verbose=False)
    Index = abs(Index)  # might get some complex values
    contrib = abs(contrib)
    Indexmean = Index.mean(axis=0)
    contribmean = contrib.mean(axis=0)
    '''
    fprintf('preselect: #PC=', training.get('estimated_num_principal_components'), 'of', d,
            fault_detection_index, 'Index=', Index, 'shape=', Index.shape, 'Index mean=', Indexmean,
            'individual contrib mean=', contribmean,
            '\nindividual contrib=', contrib, 'shape=', contrib.shape)
    '''

    contribmean_sorted_pair =  [(c,i) for i,c in sorted(zip(contribmean, range(d)), reverse=True)]
    #fprintf('contribmean_sorted_pair=', contribmean_sorted_pair)
    fprintf('Mean of individual contribution sorted:')
    for j in range(d):
        pair = contribmean_sorted_pair[j]
        fprintf('Rank = ', '%6d' % (j+1), '  variable = ', '%6d' % (np.array(pair[0])+1),
                '  contrib=', '%10.2f' % pair[1])
    # for LaTeX ...
    if useLaTex_for_documentation:
        for j in range(10):
            pair = contribmean_sorted_pair[j]
            var = np.array(pair[0])+1
            fprintf(' & ' , var, end='', file=sys.stderr)
        print('  \\\\',  file=sys.stderr)
        fprintf('\t\t& & ', end='', file=sys.stderr)
        for j in range(10):
            pair = contribmean_sorted_pair[j]
            contrib = pair[1]
            fprintf(' & ' , '%.2f' % contrib, end='', file=sys.stderr)
        fprintf('  \\\\\n\t\t[+0.8ex]', file=sys.stderr)

    from probabilistic_PCA import bool2idxContributing, bool2idxMissing
    contribmean_sorted = ((-contribmean).argsort())

    endtime = time.time()
    cputime = endtime - starttime

    #print('contribmean_sorted=', contribmean_sorted)
    #maxpre=None # DEBUG
    if maxpre is None:
        # Cutoff: Only variables with more than a certain percentage of maximum value are considered
        maxcontrib = np.amax(contribmean)

        # generate unsorted indices where contribution is high
        idx = list(np.where( contribmean_sorted >= minmaxpercentage*maxcontrib ))[0]
        #print('idx=', idx, 'type=', type(idx))
        for j in range(len(idx)):
            idx[j] = contribmean_sorted_pair[j][0]
        #print('idx=', idx, 'type=', type(idx))  ; quit()

        contributing_preselection[idx] = False
        #fprintf('contributing_preselection=', contributing_preselection, ', ')
        fprintf(fault_detection_index, 'PRESELECTION ordered at', (100*minmaxpercentage),
                '% =', str(1+np.array(idx)), '\nMissing:', bool2idxMissing(contributing_preselection))

    else:   # maximum number of selected features
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        if maxpre == 'All':
            maxpre = d
        idx = list(contribmean_sorted[:maxpre]) # conversion to list, since later used to maintain order
        #print('maxpre=', maxpre, ', idx + 1=', idx+1, 'len=', len(idx), 'type=', type(idx))
        contributing_preselection[idx] = False

        missing_preselection = np.invert(contributing_preselection) #.astype(bool)
        #print('contributing_preselection=', contributing_preselection)
        #print('missing_preselection=', missing_preselection)
        buf = str(1+np.array(idx))
        fprintf('PRESELECTION of ', len(idx), '=',  maxpre, 'highest ordered+1=', buf)
        #fprintf(d-len(idx), 'Excluded from preselection, unordered:',  bool2idxMissing(missing_preselection), '\n\n')
    
    fprintf('\nIndividual univariate contribution CPU Time=', cputime)

    fault_num = training.get('fault_num')
    fprintf('Final ordered best set of', len(idx), 'highest contributions after individual contribution search=',
            (np.array(idx)+1), 'CPU time=%.3f' % cputime, ' Fault number=', fault_num)

    #quit()
    return contributing_preselection, idx, contribmean, contribmean_sorted


def select_SBSSFS(training_data, n_components, svd_solver, fault_data, index_pre_calculated=None, fault_detection_index=None, fault_num='Undefined',
        contributing_preselection=None, presel_idx=[], k_forward=3, k_backward=2, num_elimimated_no_var=0):

    from probabilistic_PCA import PPCA_parameters, PPCA, calc_EM2_all_samples, calc_M2_one_sample, \
            bool2idxMissing, bool2idxContributing

    fprintf('----------------------------------------------------------------------------------')
    fprintf('---    S E Q U E N T I A L  S E L E C T I O N  O F  C O N T R I B U T I N G    ---')
    fprintf('----------------------------------------------------------------------------------')
    fprintf('k_forward=', k_forward, 'k_backward=', k_backward, 'num_elimimated_no_var=', num_elimimated_no_var)
    #print('contributing_preselection=\n', bool2idxContributing(contributing_preselection)) #; quit()

    n_samples, d, control_limit = PPCA_parameters(training_data)
    xmean, W, sigma2noise, C, Cinv, numeigval = PPCA(training_data, n_components=n_components, svd_solver=svd_solver, outfile=outfile)
    
    all_vars = np.ones(d, dtype=bool)
    contributing_vars = all_vars

    if not contributing_preselection is None:
        contributing_vars = contributing_preselection

    #print('index_pre_calculated=', index_pre_calculated, 'control_limit=', control_limit); quit()
    # calc statistic for a mean of several examples all variables
    #fprintf('fault_data=', fault_data, 'shape=', fault_data.shape); quit()

    '''
    if index_pre_calculated is None:
        M2 = calc_EM2_all_samples(fault_data, xmean, C, all_vars)
        #fprintf('M2 for all variables =', M2)
        fprintf('M2 for mean of ', fault_data.shape[0], 'samples and all variables = %.2f' % M2,
            'control_limit=%.2f' % control_limit)
        if M2 > control_limit:
            fprintf('\nF A U L T  D E T E C T E D: M2 = %.2f' % M2, '> %.2f' % control_limit, '===========\n')
        else:
            fprintf('\nN O  F A U L T  D E T E C T E D: M2 = %.2f' % M2, '<= %.2f' % control_limit, '===========\nReturning...\n')
            return None, None
    else:
        fprintf('\nF A U L T  D E T E C T E D: M2 pre-calculated = %.2f' % index_pre_calculated, '> %.2f' % control_limit, '===========\n')
    '''


    fprintf('-----------------------')
    fprintf('--- B A C K W A R D ---')
    fprintf('-----------------------')
    starttime = time.time()

    #print('presel_idx before BACKWARD=', presel_idx)
    fprintf('Contributing before SBS=', bool2idxContributing(contributing_vars))  #; quit()
    fprintf('Missing before SBS=', bool2idxMissing(contributing_vars))  #; quit()

    contributing = contributing_vars
    k = list(contributing).count(True)    # Number of contributing
    #k = d

    if num_elimimated_no_var > 0:
        fprintf('ADJUSTING SEQUENTIAL PARAMETERS...')
        k_forward -= num_elimimated_no_var
        fprintf('k_forward NEW=', k_forward)
        #quit()

    #fprintf('n_samples=', n_samples, 'd=', d, 'control_limit=%.2f' % control_limit)
    if k_backward >= d or k_backward < 1 or k_forward < k_backward or k_forward >= d or k <= k_backward:
        fprintf('INVALID VALUE: d=', d, 'k_forward=', k_forward, 'k_backward=', k_backward, 'Preselected=', k, '. Returning...\n\n')
        return None, None
    done = False

    while not done:
        score = np.zeros(d)
        s = 0   # index of candidate to be analysed
        for i in range(d):
            if contributing[i]:
                contributing[i] = False
                score[i] = calc_EM2_all_samples(fault_data, xmean, C, contributing)
                #fprintf('Backward: Analysis of candidate ', i+1, '... Score=', score[i])
                if s == 0:
                    minPos = i
                else:
                    if score[i] < score[minPos]:
                        minPos = i
                s += 1
                contributing[i] = True

        #fprintf('score=', score)
        if k == d:
            print_array(score, formatstr=' %.2f')
        fprintf('minPos=', minPos, 'minScore=%.2f' % score[minPos], 'limit=%.2f' % control_limit)

        contributing[minPos] = False
        presel_idx.append(minPos)
        k -= 1
        fprintf('Removing variable at minPos+1 =--- %3d' % (minPos+1), '--- from contributing. ', k, 'contributing=',
                bool2idxContributing(contributing),
                'Missing=', bool2idxMissing(contributing), 'k_backward=', k_backward)

        # done when:
        # 1) Number of contributing has decreased to a single variable
        #           O R
        # 2) Number of contributing has reached k_backward AND score is below threshold
        no_more_fault_detected = score[minPos] <= control_limit
        done = k == 1 or k <= k_backward
        #done = k == 1 or (k <= k_backward and no_more_fault_detected)

        fprintf('(k==1)=', k==1, '  (k<=k_backward)=(', k, '<=', k_backward, ')=', k<=k_backward,
                '  no more fault_detected=', no_more_fault_detected, ':', fault_detection_index, '=%.2f' % score[minPos],
                'limit=%.2f' % control_limit, '\n')
        #done = True
    #print('presel_idx after BACKWARD=', presel_idx)

    fprintf('\n---------------------')
    fprintf('--- F O R W A R D ---')
    fprintf('---------------------')
    done = False
    done = k >= k_forward

    while not done:
        fprintf('Forward: k=',  k, #'contributing=', bool2idxContributing(contributing),
                'Missing=', bool2idxMissing(contributing))
        score = np.zeros(d)
        s = 0   # index of candidate to be analysed
        for i in range(d):
            if not contributing[i]:
                contributing[i] = True
                score[i] = calc_EM2_all_samples(fault_data, xmean, C, contributing)
                #fprintf('Forward: Analysis of candidate ', i+1, '... Score=', score[i])
                if s == 0:
                    minPos = i
                else:
                    if score[i] < score[minPos]:
                        minPos = i
                s += 1
                contributing[i] = False

        #print_array(score, formatstr=' %.2f')
        #fprintf('minPos=', minPos, 'minScore=%.2f' % score[minPos], 'limit=%.2f' % control_limit)

        fault_detected = score[minPos] >= control_limit
        #if fault_detected:
        if True:
            contributing[minPos] = True
            presel_idx.remove(minPos)
            #print('presel_idx after including ', minPos, '=', presel_idx)
            k += 1
            fprintf('Including variable at minPos+1 = +++ %3d' % (minPos+1), '+++ to contributing. ',
                    'k=', k, # 'contributing=', bool2idxContributing(contributing),
                    'Missing=', bool2idxMissing(contributing))
        else:
            fprintf('FAULT: Not including variable at minPos =', minPos+1)

        # done when:
        # 1) Fault would be detected with inclusion of candidate
        #           O R
        # 2) Number of contribution has reached k_forward
        done = fault_detected or k >= k_forward
        done = k >= k_forward # DEBUG: go to k_forward

        fprintf('(k>=k_forward)=(', k, '>=', k_forward, ')=', k>=k_forward,
                '  fault_detected=', fault_detected, ': min index:', fault_detection_index, '=%.2f' % score[minPos],
                'limit=%.2f' % control_limit, '\n')

    endtime = time.time()
    cputime = endtime - starttime
    fprintf('\nSequential CPU Time=', cputime)

    #buf1 = str(bool2idxMissing(contributing))
    buf1 = str(1+np.array(presel_idx))
    buf = 'Final ordered best set after sequential SBS-SFS search=%s CPU time=%.3f' % (buf1, cputime)
    buf = buf + ' Fault number=' + fault_num
    fprintf(buf)

    fprintf('FINAL CONTRIBUTING=', bool2idxContributing(contributing), '\n\tFINAL MISSING SEQUENTIAL ORDERED=',
            buf1, 'Min fault detection index=', score[minPos] )
    return contributing, cputime



def select_b_and_b(training_data, n_components, svd_solver, fault_data, fault_detection_index=None, fault_num='Undefined', k_select=2, trace=False):

    fprintf('----------------------------------------------------------------------------------')
    fprintf('--- B R A N C H  &  B O U N D  S E L E C T I O N  O F  C O N T R I B U T I N G ---')
    fprintf('----------------------------------------------------------------------------------')

    from probabilistic_PCA import PPCA_parameters, PPCA, calc_EM2_all_samples, calc_M2_one_sample, \
            bool2idxMissing, bool2idxContributing
    n_samples, d, control_limit = PPCA_parameters(training_data)
    xmean, W, sigma2noise, C, Cinv, numeigval = PPCA(training_data, n_components=n_components, svd_solver=svd_solver, outfile=outfile)
    

    D = d
    d = k_select
    r = D   # Cardinality of Psi
    fprintf('k_select=', k_select, 'D=', D, 'd=', d, 'r=', r)
    contributing = np.ones(D, dtype=bool)

    Xi = np.array(range(D), dtype=int)     # The nested set of top down candidates for elimination
    Xd = np.zeros(d, dtype=int)            # The current best set of d features (is updated at leaf level)
    Psi = np.copy(Xi)                      # The current set of available candidates for deletion from Xi
                                           # depends on Xi and the ordered sequence of subtrees
    auxFeatSet = np.zeros(D-1, dtype=int)

    def b_and_b( level, B, leaf_visits, D, d, r, Xi, Xd, Psi, auxFeatSet, trace=False ):
        
        def delFeat(featset, feat):
            '''Eliminate a feature from a array-like set by downshift
            '''
            #fprintf('delFeat> featset=\n', featset, 'feat=', feat)
            lenset = len(featset)
            i = 0; found = False
            while not found and i < lenset:
                found = featset[i] == feat
                if not found:
                    i += 1
            assert found, 'Could not find element in set'
            for j in range(i,lenset-1):
                featset[j] = featset[j+1]
            return featset

        # diff diffSets( Psi, Q_seq, r, q );
        def diffSets(set1, set2):
            '''delete all elements from set1 that are in set2:
            Assumptions: 1) len1 >= len2, 2) all elements in set2 appear in set1
            '''
            
            for i in range(len(set2)):
                set1 = delFeat(set1, set2[i])
            return set1


        # calculate q: number of possible subtrees
        q = r - (D-d-level-1)

        next_level_is_leaf = level+1 == D-d; # equivalent to q == r

        if trace:
            fprintf('B&B: Current next level=', level+1, ' --- Leaf level=', D-d, 'Number of possible subtrees=', q)

        # determine the criterion for all potential features to be discarded
        # all potential features to be discarded are in Psi
        # the succeeding level might be a leaf or not

        if trace:
            fprintf('# of available features =', r, 'Xi[level=', level, ']=', feats(Xi[:D-level]), 'Psi=', feats(Psi[:r]))

        aux_order = np.zeros(shape=(r,2))   # Set Psi with feature number and selection criterion
        for j in range(r):
            contributing.fill(True)
            # generate the potential feature set: eliminate the subtree candidate from Xi
            cardAux = 0
            for p in range(D-level):
                var = Xi[p]
                if Psi[j] != var:
                    auxFeatSet[cardAux] = var
                    cardAux += 1
                    contributing[int(var)] = False
            if trace:
                fprintf('$$$$$$ Calculating selection criterion from set+1=', feats(auxFeatSet[:cardAux]), end='')

            # contributing from auxFeatSet
            if trace:
                fprintf('\tMissing=', bool2idxMissing(contributing));

            J = calc_EM2_all_samples(fault_data, xmean, C, contributing)
            aux_order[j][0] = Psi[j]
            aux_order[j][1] = J
            if trace:
                fprintf('{left_out=', 1+int(Psi[j]), 'J=', J, 'Psi=', feats(Psi[:r]))

        aux_order = sorted(zip(aux_order[:,1],aux_order[:,0]), reverse=True)
        if trace:
            fprintf('Psi sorted_by_J raw=', aux_order)
        for p in range(r):
            Psi[p] = int(aux_order[p][1])
        if trace:
            fprintf('Psi sorted_by_J=', feats(Psi[:r]))

        Q_seq = np.copy(Psi[:q])
        Q_seq_crit = np.array(aux_order)[:q,0]
        if trace:
            fprintf('Q_seq=', feats(Q_seq), 'Q_seq_crit=', Q_seq_crit, 'q=', q, 'r=', r)

        Psi = diffSets(Psi, Q_seq)

        r = r - q
        if trace:
            fprintf('--- New Psi with', r, 'feats =', feats(Psi[:r]))  #; quit()

        # first process the subtrees of the node that had highest criterion: right to left
        # for all possible subtrees (might be leaf of not)
        #for i in range(q):
        for i in range(q-1,-1,-1):  # generates inverted sequence [q-1, ..., 0]
            J_Q, feat_Q = Q_seq_crit[i], Q_seq[i]
            if trace:
                fprintf('====== Level=', level, 'Q_seq[subtree=%2d]=(left_out+1=%2d,%12.5f) Bound=%12.5f'
                    %(i, feat_Q+1, J_Q, B) )

            if next_level_is_leaf:
                dummy = -1
                '''
                auxset = np.copy(Xi[:d+1])
                auxset = delFeat(auxset, feat_Q)
                fprintf('--> Checking leaf=', feats(auxset[:d]), 'J=', J_Q, 'Bound=', B)
                '''
            # check bound
            if J_Q <= B:
                if trace:
                    fprintf('BOUND: ', J_Q, '<=', B)
                    # remove feature of subtree from top-down nested set of remaining features Xi
                    fprintf('**** Removing Q_seq[',i,']=', Q_seq[i], 'from Xi=', feats(Xi[:D-level]))
                Xi = delFeat(Xi, feat_Q)
                if trace:
                    fprintf('Remaining features Xi=', feats(Xi[:D-level-1]))
                if next_level_is_leaf:
                    # successor i of current node is leaf and surpassed bound
                    # Bound updating and best feature set updating
                    B = J_Q
                    Xd = np.copy(Xi[:d])
                    #if trace:
                    fprintf('Next level is leaf: NEW BOUND = %12.5f' % B,
                            ' --- Best set selected so far: ', feats(Xd), '<=== *****')
                    outfile.flush()
                else:
                    # successor i of current node is no leaf but criterion value is still higher than bound
                    # ==> have to explore subtree  R E C U R S I O N =====================
                    if trace:
                        fprintf('R E C U R S I O N  T O  L E V E L', level+1, 'with new r=', r,
                                'Xi=', feats(Xi[:D-level-1]))
                    #input('Press Enter to continue into recursion level %s ...' % str(level+1))
                    Xi, Xd, Psi, r, B, leaf_visits = \
                        b_and_b( level+1, B, leaf_visits, D, d, r, Xi, Xd, Psi, auxFeatSet, trace )

                # Put the eliminated candidate back to Xi
                if trace:
                    fprintf('Xi before putting candidate Q_seq[',i,']=', Q_seq[i], ' back=', feats(Xi[:D-level-1]))
                Xi[D-level-1] = Q_seq[i]
                if trace:
                    fprintf('Xi after putting candidate back=', feats(Xi[:D-level])) #; quit()
            else:
                if trace:
                    fprintf('--- %12.5f does not meet bound %12.5f for subtree %2d at level %2d!!!' %
                        (J_Q, B, i+1, level), 'Q_seq=', feats(Q_seq))
            '''
            '''

            if next_level_is_leaf:
                leaf_visits += 1
                fprintf(' *** Leaf # %.0lf visited -- Up from leaf level %d' % (leaf_visits, D-d) )

            # put the feature of the explored branch back to the set of available features Psi
            Psi[r] = Q_seq[i]
            r += 1

        #fprintf('Backtrack from level', level)
        return Xi, Xd, Psi, r, B, leaf_visits

    starttime = time.time()
    Xi, Xd, Psi, r, B, leaf_visits = b_and_b( 0, np.inf, 0, D, d, r, Xi, Xd, Psi, auxFeatSet, trace )
    best_set = Xd

    endtime = time.time()
    cputime = endtime - starttime
    fprintf('\nB&B        CPU Time=', cputime)

    contributing.fill(True)
    contributing[Xd] = False
    fprintf('FINAL CONTRIBUTING=', bool2idxContributing(contributing), '\n\tFINAL MISSING B&B ORDERED=', feats(Xd))
    buf = 'Final unorderd best set after b&b with bound %.2f=%s CPU time=%.3f' % (B, best_set+1, cputime)
    buf = buf + ' Fault number=' + fault_num
    fprintf(buf)
    return contributing, cputime


def select_exhaustive(training_data, n_components, svd_solver, fault_data, index_pre_calculated=None,
        fault_detection_index=None, fault_num='Undefined', k_select=2):

    fprintf('----------------------------------------------------------------------------------')
    fprintf('--- E X H A U S T I V E  S E L E C T I O N  O F  C O N T R I B U T I N G ---')
    fprintf('----------------------------------------------------------------------------------')
    #fprintf('k_select=', k_select)
    M2 = index_pre_calculated

    from probabilistic_PCA import PPCA_parameters, PPCA, calc_EM2_all_samples, calc_M2_one_sample, \
            bool2idxMissing, bool2idxContributing
    n_samples, d, control_limit = PPCA_parameters(training_data)
    xmean, W, sigma2noise, C, Cinv, numeigval = PPCA(training_data, n_components=n_components, svd_solver=svd_solver, outfile=outfile)
    
    all_vars = np.ones(d, dtype=bool)
    contributing_vars = all_vars

    '''
    if index_pre_calculated is None:
        M2 = calc_EM2_all_samples(fault_data, xmean, C, all_vars)
        #fprintf('M2 for all variables =', M2)
        fprintf('M2 for mean of ', fault_data.shape[0], 'samples and all variables = %.2f' % M2,
            'control_limit=%.2f' % control_limit)
        if M2 > control_limit:
            fprintf('\nF A U L T  D E T E C T E D: M2 = %.2f' % M2, '> %.2f' % control_limit, '===========\n')
        else:
            fprintf('\nN O  F A U L T  D E T E C T E D: M2 = %.2f' % M2, '<= %.2f' % control_limit, '===========\nReturning...\n')
            return None, None
    else:
        M2 = index_pre_calculated
        fprintf('\nF A U L T  D E T E C T E D: M2 pre-calculated = %.2f' % M2, '> %.2f' % control_limit, '===========\n')
    '''


    fprintf('\n\n=========================================================')
    fprintf('=== E X H A U S T I V E  S E A R C H  ', k_select, 'from', d, '   ===')
    fprintf('=========================================================')
    fprintf('n_samples=', n_samples, 'd=', d, 'control_limit=%.2f' % control_limit)
    if k_select >= d or k_select < 1:
        fprintf('INVALID VALUE: d=', d, 'k_select=', k_select, '. Returning...\n\n')
        return None, None

    done = False
    contributing = np.copy(contributing_vars)
    k = list(contributing).count(True)    # Number of contributing


    from itertools import combinations
    from scipy.special import binom
    numfeat = k

    fprintf('Preselected=',  k, 'of a total of', d, ' --- contributing=',
                bool2idxContributing(contributing), '\n\tMissing=', bool2idxMissing(contributing))

    num_combinations = int(binom(numfeat, k_select))
    fprintf('Exhaustive ', k_select, ' from ', numfeat, ', # of combinations = ', numfeat,
             '!/(', k_select, '!', '(', numfeat, '-',  k_select, ')!) = ', num_combinations, sep='')
    c = list(combinations(range(numfeat), k_select))
    max_show = 500
    if num_combinations < max_show:
        fprintf(c)
    #fprintf('fault_data=', fault_data, 'shape=', fault_data.shape)
    #quit()

    J = np.zeros(num_combinations)
    minJ = M2
    starttime = time.time()

    fprintf('Exhaustive search...', file=sys.stderr)
    best_cand = ()
    for j in range(num_combinations):
        cand = c[j]
        contributing = np.copy(contributing_vars)
        for p in range(len(cand)):
            contributing[cand[p]] = False
            #fprintf('\t\t', cand[p])

        # Calculate criterion with the contributing variables
        J[j] = calc_EM2_all_samples(fault_data, xmean, C, contributing)
        if J[j] < minJ:
            minJ = J[j]
            best_cand = cand
        if outfile == sys.stdout:
            fprintf('\tJ[', '%6d' % (j+1), 'of %6d' % num_combinations, ']=', '%15.5f' % J[j],
                '  cand+1= ', np.array(cand)+1, 'Best: J=', '%15.5f' % minJ,
                '  best_cand+1= ', np.array(best_cand)+1, '\r', end='', file=sys.stderr)
            sys.stderr.flush()
        #fprintf('\tJ[', '%6d' % j, ']=', '%10.2f' % J[j], '  cand+1= ', np.array(cand)+1,
        #        ': contributing=', contributing, sep='')
        #fprintf('\ncand=', cand, 'missing=', bool2idxMissing(contributing)); quit()
    if outfile == sys.stdout:
        fprintf('',file=sys.stderr)
    fprintf('Sorting...')
    sorted_by_J = [(y, x) for y,x in sorted(zip(J,c))]
    #fprintf('sorted_by_J=', sorted_by_J)
    for j in range(min(max_show,num_combinations)):
        pair = sorted_by_J[j]
        J = pair[0]
        fprintf('Rank = %6d' % (j+1), '  J=%12.4f' % J, 'missing=', np.array(pair[1])+1,
                'Below control limit %12.4f' % control_limit, '=', J < control_limit)

    best_set = (sorted_by_J[0])[1]
    #fprintf('Best missing indices =', best_set)

    contributing = np.copy(all_vars)
    for i in range(len(best_set)):
        contributing[best_set[i]] = False

    buf1 = str(np.array(best_set)+1)
    fprintf('Exhaustive: FINAL CONTRIBUTING=', bool2idxContributing(contributing),
            '\n\tFINAL MISSING EXHAUSTIVE=',  buf1, 'Min fault detection index=', sorted_by_J[0][0] )
    endtime = time.time()
    cputime = endtime - starttime
    fprintf('\nExhaustive CPU Time=', cputime)

    buf = 'Final unorderd best set after exhaustive search=%s CPU time=%.3f' % (buf1, cputime)
    buf = buf + ' Fault number=' + fault_num
    fprintf(buf)

    return contributing, cputime


def standardize(training, data, fault_detection_index):
    fprintf('Standardizing data. fault_detection_index=', fault_detection_index)
    meanX = training.get('meanX')
    if fault_detection_index == 'M2':
        fprintf('Standardize: Special non-standardized data. Only centralizing data ...')
        data = data - meanX
    else:
        stdX = training.get('stdX')
        data = (data - meanX) / stdX
    return data


def elim_no_variance(training_data, fault_data, mask):
    epsvar = 1e-10  # Below this = no variance considered
    train_var = np.var( training_data, axis=0 )
    idx_train = train_var >= epsvar
    test_var = np.var( fault_data, axis=0 )
    idx_test = test_var >= epsvar
    idx = np.logical_and(idx_train, idx_test)
    num_elimimated = list(idx).count(False)
    if num_elimimated > 0:
        fprintf('Preprocessing: Eliminating ', num_elimimated, 'variables without variance =', 1+np.where(idx-1)[0])
    #fprintf('Variance training=\n', train_var)
    #fprintf('\nVariance test=\n', test_var)
    #fprintf('\nidx_train=', idx_train, '\nidx_test=', idx_test, '\nAND=\n', idx)
    #fprintf('\nOld mask=\n', mask, 'type=', type(mask))

    idx = list(np.where( idx )[0])
    #fprintf('\nidx=', idx, 'type=', type(idx))
    newmask = list(set.intersection(set(mask), set(idx)))
    #fprintf('\nnewmask=\n', newmask, 'type=', type(newmask))
    return newmask, num_elimimated


# global variables
benchmark = 'pubBnB'
benchmark = 'TE_Ding'
benchmark = 'b_and_b_test'
benchmark = 'pub1'
benchmark = 'TE'

def TE_instantiate():
    fprintf('DATA: Tennessee Eastman')
    rootdir = '/home/thomas/Dropbox/software/TE/Tennessee_Eastman/'
    srcdir = rootdir + 'te/simulator/'
    sys.path.append(srcdir)
    datadir = rootdir + 'TE_process/data/'
    from TE import TE
    te = TE()
    return te, datadir


def main():
    #fprintf('Executing main() ....')
    featname = training_labels = fault_labels = None
    mask = list(range(52))
    fault_detection_index = 'M2'    # SPE  T2  Combined  M2
    maxpre = None
    minmaxpercentage = 0.0
    fault_start=0
    fault_stop=None
    fprintf('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

    if benchmark == 'b_and_b_test':
        fprintf('DATA: Branch and Bound Debug')
        training_data, fault_data = simulateX_pubBnB(use_only_four_vars=True)
        n_components = 2
        window = 1
        fault_num = 'fault'
        k_backward = 2
        k_forward = 3
        k_select = 2    # exhausive search and b&b


    if benchmark == 'pub1':
        fprintf('DATA: Qin paper')
        window = 30
        # Generate training and test data by simulation

        #faultdirections = 'Random'
        #training_data, training_labels, fault_data, fault_labels = simulateX_pub1(faultdirections)

        faultdirections = (1,4)
        training_data, training_labels, fault_data, fault_labels = simulateX_pub1(faultdirections)

        fault_num='fault'
        n_components = 2
        k_backward = 2  # How many should contribute after this selection stage 
        k_forward = 3   # How many should contribute after this selection stage
        k_select = 3    # exhausive search and b&b
        fprintf('=================\n\tBenchmark: Qin paper\n==================\n')
        fprintf('window=',window,'faultdirections=',faultdirections,'num_principal_components=',n_components,
                '\n\tmaxpre=',maxpre,'k_backward=',k_backward,'k_forward=',k_forward,'k_select=',k_select)


    if benchmark == 'pubBnB':
        fprintf('DATA: Branch & Bound paper')
        window = 1
        # Generate training and test data by simulation
        training_data, fault_data = simulateX_pubBnB(use_only_four_vars=False)
        fault_num='fault'
        n_components = 2
        k_backward = 2
        k_forward = 4
        k_select = 2    # exhausive search and b&b
        

    ### T E ###
    if benchmark == 'TE' or benchmark == 'TE_Ding':
        fprintf('DATA: Tennessee Eastman')
        te, datadir = TE_instantiate()
        
        # # # # # # # 
        fault_num='01'
        if not faultarg is None:
            fault_num = faultarg

# xxx
        svd_solver = 'full'

        n_components = 'mle'
        n_components = 0.99
        n_components = 7

        maxpre = None; minmaxpercentage = 0.5
        maxpre = 10; minmaxpercentage = 0.0
        presel_idx = []
        if maxpre is None:
            assert minmaxpercentage > 0.0 and minmaxpercentage <= 1.0, 'invalid minmaxpercentage'
            k_backward = max(2, int(minmaxpercentage/100.0 * te.numfeat))
        else:
            k_backward = max(2, te.numfeat-2*maxpre)   # How many should contribute after this selection stage 

        k_select = 2     # exhausive search and b&b
        # Ignore 'Agitator speed which is constant
        k_forward = te.numfeat - 1 - k_select # How many should contribute after this selection stage
        #print('te.numfeat=', te.numfeat, 'k_forward=', k_forward); quit()

        discard_first_train = None

        discard_first_test = None  # Use all samples from the test data
        discard_first_test = 160

        fprintf('Number of samples ignored at beginning of test data = ', discard_first_test)

        #discard_upto_test = None 
        window = 30

        standardize = False

        # # # # # # # 
        #fprintf('Reading TE train and test pair from dir\n', datadir, '\nfault_num=', fault_num, 'standardize=',standardize)
        te.read_train_test_pair(datadir=datadir, fault_num=fault_num, standardize=standardize)

        training_data, fault_data = te.Xtrain, te.Xtest

        #print('training_data=\n', training_data, 'shape=', training_data.shape)
        #print('fault_data=\n', fault_data, 'shape=', fault_data.shape)  ; quit()

        training_data = training_data[discard_first_train:,:]   # Discard the first samples

        fault_start=0
        fault_stop=None

        featname = te.featname

        fprintf('TE fault number =', fault_num)
        if benchmark == 'TE_Ding':
            fprintf('DATA: Tennessee Eastman, Ding paper')
            # [5]
            n_components = 7    # [5], sec. 4.1
            mask = range(22) + range(22+19, 52)
            assert len(mask) > n_components, 'Less features than principal components'
            k_backward = 18
            k_forward = 20

            training_data, featname = te.filter_vars(training_data, mask)
            fault_data, featname = te.filter_vars(fault_data, mask)
            #fprintf('filtered features=', featname); quit()

        # Filter out variables that have no variance either in training or test
        newmask, num_elimimated_no_var = elim_no_variance(training_data, fault_data, mask)
        training_data, featname = te.filter_vars(training_data, newmask)
        fault_data, featname = te.filter_vars(fault_data, newmask)
        mask = newmask
        #print('After no variance filter: num_variables train=', training_data.shape[1],
        #    'num_variables test=', fault_data.shape[1], 'No var=', num_elimimated_no_var); quit()


    #fprintf('training_data=\n', training_data, '\ntraining_labels=\n', training_labels,
    #        '\nfault_data=\n', fault_data, '\nfault_labels=\n', fault_labels);# quit()

    '''
    if not no_graph:
        te.signal_plot(infile=None, X=training_data, dropfigfile='/tmp/outfigtrain.svg',
                title='Training Data', mask=mask)
        te.signal_plot(infile=None, X=fault_data, dropfigfile='/tmp/outfigtest.svg',
                title='Test Data', mask=mask)
    '''
    fprintf('\n((((---  E X P E R I M E N T A L  P A R A M E T E R S ---))))')
    fprintf('benchmark=', benchmark)
    fprintf('fault_detection_index=', fault_detection_index)
    #fprintf('featname=', featname)
    fprintf('Eliminated no variance=', num_elimimated_no_var)
    fprintf('mask[size=',len(mask),']=', mask)
    fprintf('fault_detection_index=', fault_detection_index)
    fprintf('maxpre=', maxpre)
    fprintf('minmaxpercentage=', minmaxpercentage)
    fprintf('num_principal_components=', n_components)
    fprintf('window=', window)
    fprintf('fault_num=', fault_num)
    fprintf('k_backward=', k_backward)
    fprintf('k_forward=', k_forward)
    fprintf('k_select=', k_select)
    fprintf('fault_start=', fault_start)
    fprintf('fault_stop=', fault_stop)
    if benchmark == 'pub1':
        fprintf('faultdirections=', faultdirections)
    if benchmark == 'TE' or benchmark == 'TE_Ding':
        #fprintf('rootdir=', rootdir)
        #fprintf('srcdir=', srcdir)
        fprintf('datadir=', datadir)
        fprintf('discard_first_train=', discard_first_train)
        fprintf('discard_first_test=', discard_first_test)
        #fprintf('discard_upto_test=', discard_upto_test)
    fprintf('(((((((((((((((------------------)))))))))))))))\n')


    semilogy=True
    semilogy=False

    # Generation of all training parameters
    verbose = False
    training = gen_train(training_data, fault_data, n_components=n_components, fault_num=fault_num, featname=featname, verbose=verbose)

    #from plot_2_D_statistics import plot_2_D_statistics
    #plot_2_D_statistics(training); quit()
    #calc_all_fault_detection_indices(training)
    #detect_contrib(training)
    #quit()

    # F A U L T  D E T E C T I O N
    is_fault, mean_index_win, pos = detect_fault(training, fault_detection_index, fault_data,
            discard_first_test=discard_first_test, window=window)

    fprintf('Detection: Fault=', is_fault, ', mean index of ', window, 'samples=', mean_index_win,
            'pos=', pos, 'of', fault_data.shape[0],
            'samples. Discarded fault samples at beginning=', discard_first_test)

    # for LaTeX ...
    if useLaTex_for_documentation:
        print('IDV (', fault_num, ') & ', pos, ' & %.2f' % mean_index_win, end='', file=sys.stderr)

    if is_fault:
        # Set fault data only to the analyzed window
        fault_data_window = fault_data[pos:pos+window]

        '''
        print('fault_data_window=', fault_data_window, 'shape=', fault_data_window.shape)
        mask = [-1+ 7, -1+ 39]
        te.signal_plot(infile=None, X=fault_data, divide_by_mean=False, subtract_mean=False,
            standardize=True, mask=mask)
        te.signal_plot(infile=None, X=fault_data_window, divide_by_mean=False, subtract_mean=False,
            standardize=True, mask=mask)
        '''
        # takes the contribution only of the window in which the fault was detected
        if not no_graph:
            #print('no_graph=', no_graph)
            # the contribution plot takes the contribution of the whole test data set
            contribution_and_time_plot(training, fault_data, discard_first_test=discard_first_test,
                    fault_start=pos, window=window,
                    fault_detection_index=fault_detection_index,
                    semilogy=semilogy, benchmark=benchmark, dropfigdir=dropfigdir)

            '''
            contribution_and_time_plot(training, fault_data_window, discard_first_test=discard_first_test, fault_start=pos, fault_stop=pos+window,
                    fault_detection_index=fault_detection_index,
                    semilogy=semilogy, benchmark=benchmark, dropfigdir=dropfigdir)
            '''

        # P R E S E L E C T I O N
        # only used in the sequential search algorithm
        contributing_preselection, presel_idx, contribmean, contribmean_sorted = preselect(training, fault_data_window,
                fault_detection_index=fault_detection_index, maxpre=maxpre, minmaxpercentage=minmaxpercentage)
        #preselection = None; fprintf('FORCING NO PRESELECTION...'); # No preselection

        '''
        fprintf('\n\nFAULT DETECTION\nINDEX = ', fault_detection_index)
        if fault_detection_index != 'M2':
            fprintf('\t#PC=', training.get('n_components'))

        te.signal_plot(infile=None, X=fault_data, divide_by_mean=False, subtract_mean=True,
            standardize=False, dropfigfile='/tmp/outfigtest.svg',
            title='Test Data', mask=[presel_idx[0], presel_idx[1], presel_idx[2], presel_idx[3]])
        quit()
        '''


        from probabilistic_PCA import bool2idxContributing, bool2idxMissing
        #te.signal_plot(infile=None, X=fault_data_window, mask=bool2idxMissing(preselection))


        if useLaTex_for_documentation or not no_graph:
            quit()
        # yyy

        do_SBSSFS = False
        do_SBSSFS = True

        do_b_and_b = False
        do_b_and_b = True

        do_exhaustive = False
        do_exhaustive = True
        fprintf('Fault number=', fault_num, 'k_select=', k_select, 'k_backward=', k_backward, 'k_forward=', k_forward)

        if do_SBSSFS:
            contributing, cputime = select_SBSSFS(training_data, n_components, svd_solver, fault_data_window, mean_index_win,
                fault_detection_index, fault_num,
                contributing_preselection=contributing_preselection, presel_idx=presel_idx,
                k_forward=k_forward, k_backward=k_backward, num_elimimated_no_var=num_elimimated_no_var)
        outfile.flush()

        if do_exhaustive:
            contributing, cputime = select_exhaustive(training_data, n_components, svd_solver, fault_data_window, mean_index_win,
                    fault_detection_index, fault_num, k_select=k_select)
        outfile.flush()

        if do_b_and_b:
            contributing, cputime  = select_b_and_b(training_data, n_components, svd_solver, fault_data_window,
                    fault_detection_index, fault_num, k_select=k_select, trace=False)
        outfile.flush()
    else:
        if not no_graph:
            contribution_and_time_plot(training, fault_data, discard_first_test=discard_first_test, fault_start=pos, window=window,
                fault_detection_index=fault_detection_index, semilogy=semilogy, benchmark=benchmark, dropfigdir=None)



def test_plot_condition():
    fprintf('>>>> U S I N G  I R I S  D A T A <<<<<\n')
    import sklearn.datasets as datasets
    # Get iris data
    iris = datasets.load_iris()
    X = iris.data
    #fprintf('Iris data: X=\n', X)
    y = iris.target
    classlabel = np.unique(iris.target)
    classname = iris.target_names
    featname = iris.feature_names
    from TE import TE
    te = TE()
    te.plot_condition(X, y, classlabel, classname, featname, plot_time_axis=True, title='Iris data')


def test_exhaustive():
    from itertools import combinations
    from scipy.special import binom
    numfeat = d = 5
    all_vars = np.ones(d, dtype=bool)
    preselection = all_vars

    maxsel = 3
    for i in range(2,maxsel+1):
        fprintf('Exhaustive ', i, 'from', numfeat,
                '# of combinations=(', numfeat, '!) / (', i, '!', '(', numfeat, '-',  i, ')!) = ',
                '%d' % binom(numfeat, i))
        c = list(combinations(range(numfeat), i))
        fprintf(c)
        for j in range(len(c)):
            cand = c[j]
            contributing_vars = np.copy(preselection)
            for k in range(len(cand)):
                contributing_vars[cand[k]] = False
                #fprintf('\t\t', cand[k])
            fprintf('\tcand=', cand, ': contributing_vars=', contributing_vars)


def pub1_export_Matlab():
    from scipy.io import savemat

    faultdirections = 'Random'
    trainname = 'training_data_Random'
    faultname = 'fault_data_Random'
    outfile = '/home/thomas/tmp/pubQin_Random.mat'
    training_data, training_labels, fault_data, fault_labels = simulateX_pub1(faultdirections)

    '''
    faultdirections=(1,4)
    trainname = 'training_data__1_4'
    faultname = 'fault_data__1_4'
    outfile = '/home/thomas/tmp/pubQin_directions__1_4.mat'
    training_data, training_labels, fault_data, fault_labels = simulateX_pub1(faultdirections)
    '''

    fprintf('Exporting training data and test data from Qin paper to Matlab .mat file', outfile,
            '\n\tFault directions=', faultdirections)
    savemat(outfile, mdict={trainname: training_data, faultname: fault_data})


def pub1_import_Matlab():
    from scipy.io import loadmat
    matfilename = '/home/thomas/Downloads/thomas_1'
    mat = loadmat(matfilename)
    fprintf('Matlab file', matfilename, 'loaded:', mat)
    DD = mat.get('DD'); fprintf('DD=\n', DD, 'shape=', DD.shape); # DD = DD.T;
    x = mat.get('x'); fprintf('x=\n', x, 'shape=', x.shape)
    training_labels = np.ones(x.shape[0])
    xf = mat.get('xf'); fprintf('xf=\n', xf, 'shape=', xf.shape)
    test_labels = DD
    return x, training_labels, xf, test_labels

def pub1_tab5():
    #training_data, training_labels, fault_data, fault_labels = simulateX_pub1('Random')

    training_data, training_labels, fault_data, fault_labels = pub1_import_Matlab()

    training = gen_train(training_data, fault_data, n_components=4, fault_num=None)
    fprintf('num_principal_components=', training.get('n_components'))
    calc_all_fault_detection_indices(training)
    fprintf('fault_labels=', fault_labels)
    numfaults = fault_labels.shape[0]
    for i in range(numfaults):
        x = fault_data[i]
        # fprintf('fault', i+1, '=', fault_labels[i])

def classification_experiment(fault_num, selected):
    discard_first_train = 0
    discard_first_test = 160
    discard_upto_test = None

    te, datadir = TE_instantiate()

    fprintf('Reading TE train and test pair from dir\n', datadir, '\nfault_num=', fault_num, 'standardize=',standardize)
    te.read_train_test_pair(datadir=datadir, fault_num=fault_num, standardize=True)
    #training_data, fault_data = te.Xtrain, te.Xtest
    training_data, fault_data = te.Xstandardized_train, te.Xstandardized_test

    training_data = training_data[discard_first_train:,selected]   # Discard the first samples

    # Test data 48h with 960 samples = 20 samples/h = 1 sample each 3 min --- Fault after 8h = 160 samples
    fault_data = fault_data[discard_first_test:discard_upto_test,selected]

    n_train = training_data.shape[0]
    n_test = fault_data.shape[0]
    training_labels = np.ones(n_train, dtype=int)
    fault_labels = -np.ones(n_test, dtype=int)
    
    
    featname = te.featname
    classname = np.array(['normal', 'fault'], dtype='|S7')
    fprintf('TE fault number =', fault_num)
    
    maxshow = 10
    X = np.concatenate([training_data, fault_data], axis=0)
    y = np.concatenate([training_labels, fault_labels], axis=0)
    #fprintf('X=', X[:maxshow])    ; quit()
    '''
    fprintf('X=\n', X, 'shape=', X.shape, '\ny=\n', y)
    fprintf('training_data=\n', training_data[:maxshow], 'shape=', training_data.shape, '\ntraining_labels=', training_labels[:maxshow])
    fprintf('fault_data=\n', fault_data[:maxshow], 'shape=', fault_data.shape, '\nfault_labels=', fault_labels[:maxshow])
    '''

        
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
    
    #clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    #clf = QuadraticDiscriminantAnalysis()
    #clf = LinearDiscriminantAnalysis()
    clf = KNeighborsClassifier(n_neighbors=1)
    
    
    y_pred_overall = []
    y_test_overall = []
    n_splits = 10;
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)

    from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
    do_select = False
    do_select = True
    D = 52
    nums = np.array(range(D), dtype=int)

    '''
    if do_select:
        selector = SelectKBest(mutual_info_classif, k=1).fit(X, y)
        fprintf('selector.scores_=', selector.scores_)
        fprintf('selector.pvalues_=', selector.pvalues_)
        X = np.copy(selector.transform(X))
        X = SelectKBest(mutual_info_classif, k=10).fit_transform(X, y)
    '''
    
    i = 0
    for train_index, test_index in skf.split(X, y):
        #fprintf('Fold #', i+1, 'de', n_splits, '\nTRAIN:', train_index, '\nTEST:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if do_select:
            k = 7
            selector = SelectKBest(mutual_info_classif, k=k).fit(X_train, y_train)
            scores = selector.scores_
            selected = selector.get_support()
            aux_order = sorted(zip(scores,nums), reverse=True)
            fprintf('aux_order=',aux_order)   #; quit()
            aux_order = aux_order[:k]
            fprintf('\naux_order cropped to', k, '=',aux_order)   #; quit()

            fprintf('selector.scores_=', scores, 'shape=', scores.shape)
            fprintf('selector.pvalues_=', selector.pvalues_)
            fprintf('selector.get_params()=', selector.get_params())
            fprintf('selector.get_support()=', selected)
            fprintf('Feat num=', nums[selected])
            fprintf('Feat scores=', scores[selected])
            X_train = np.copy(selector.transform(X_train))
            X_test = np.copy(selector.transform(X_test))
            #X_train = np.copy(SelectKBest(mutual_info_classif, k=2).fit_transform(X_train, y))
        '''
        '''
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        i += 1
        #fprintf('y_test=', y_test, '\ny_pred=', y_pred)

        y_pred_overall = np.concatenate([y_pred_overall, y_pred])
        y_test_overall = np.concatenate([y_test_overall, y_test])

    fprintf('y_test_overall=', y_test_overall, '\ny_pred_overall=', y_pred_overall)
    print ('Classification Report: ')
    print (classification_report(y_test_overall, y_pred_overall, target_names=classname, digits=3))
    print ('Accuracy=', '%.2f %%' % (100*accuracy_score(y_test_overall, y_pred_overall)))
    print ('Confusion Matrix: ')
    print (confusion_matrix(y_test_overall, y_pred_overall))


def test_classification_experiment():
    fault_num = '09'
    selected = np.zeros(52, dtype=bool)      # select none
    selected[2] = True
    selected[22] = True
    selected = np.ones(52, dtype=bool)      # select all
    classification_experiment(fault_num, selected)


if __name__ == "__main__":
    fprintf('Main start');
    if useLaTex_for_documentation:
        stderrfile = './tmp.txt'
        print('Redirecting stderr to ', stderrfile, '...')
        sys.stderr = open(stderrfile, 'a')

    #test_classification_experiment(); quit()
    #pub1_tab5(); quit()
    #pub1_export_Matlab(); quit()
    #test_plot_condition(); quit()
    #test_exhaustive(); #quit()

    #fprintf('sys.argv=', sys.argv, 'with length=', len(sys.argv))
    #progname = sys.argv[0]
    if len(sys.argv) > 1:
        for i in range(1,len(sys.argv)):
            arg = sys.argv[i]
            #print('argv[',i,']=', arg, 'arg[0:2]=', arg[0:2], 'arg[2:4]=', arg[2:4])
            if arg[0:2] == '-n':
                no_graph = True
                print('Disable graphic output...')
            if arg[0:2] == '-f':
                faultarg = arg[2:4]
                print('Reading fault number as parameter. Fault number=', faultarg)

    #quit();
    #main()

    import timeit
    t = timeit.timeit("main()", setup="from __main__ import main", number=1)
    fprintf('Total time=', t)

    fprintf('Main stop');
    if outfile != sys.stdout:
        print('=== Closing log file', logfname)
        logfile.close()
