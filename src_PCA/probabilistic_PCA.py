from __future__ import print_function

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# https://scikit-learn.org/stable/modules/decomposition.html#pca

'''
[1] Journal of Process Control, Volume 20, Issue 10, December 2010, Pages 1198-1206
    A Branch and Bound Method for Isolation of Faulty Variables through Missing Variable Analysis
    Vinay Kariwala , Pabara-Ebiere Odiowei , Yi Cao and Tao Chen

[2] Chen, Tao, and Yue Sun. "Probabilistic contribution analysis for statistical process
    monitoring: A missing variable approach." Control Engineering Practice 17.4 (2009): 469-477.

[3] Reconstruction-based contribution for process monitoring,
    Carlos F. Alcala, S. Joe Qin
    Automatica 45 (2009), pp. 1593-1600
'''

import sys
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.stats import chi2
from time import sleep

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

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


def show_eig(X, pca):
    n_samples = X.shape[0]
    # Center the data and compute the sample covariance matrix.
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
    print('cov_matrix=\n', cov_matrix)    
    eigenvalues = pca.explained_variance_
    for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
        print('eigenvector=', eigenvector)    
        print('phi\'*S*phi=', np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
        print('eigenvalue=', eigenvalue)    



# T2 statistic
def calc_T2_D_mat(X):
    S = np.cov(X, rowvar=False)
    Lambda, Phi = eigen(S)
    num_principal_components = S.shape[0]
    l = num_principal_components
    P = Phi[:, :l]
    L = np.diag(Lambda[:l])
    LInv = np.diag((1.0/Lambda[:l]))
    D = np.dot(np.dot(P, LInv), P.T)
    #print('Matrix D = PL^-1P''=\n', D)
    return D


def show_result(pca):
    print('pca.svd_solver=', pca.svd_solver)  
    print('pca.n_components_=', pca.n_components_, 'of a total of', pca.components_.shape[1], 'components')  
    print('pca.components_(eigvecs)=\n', pca.components_, 'shape=', pca.components_.shape)  
    print('pca.explained_variance_ratio_=\n', pca.explained_variance_ratio_)  
    print('pca.singular_values_(eigvals)=\n', pca.singular_values_)
    print('pca.mean_=\n', pca.mean_, 'shape=', pca.mean_.shape)
    print('pca.noise_variance_=', pca.noise_variance_)
    #print('\n')  


def TE_instantiate():
    print('DATA: Tennessee Eastman')
    rootdir = '/home/thomas/Dropbox/software/TE/Tennessee_Eastman/'
    srcdir = rootdir + 'te/simulator/'
    sys.path.append(srcdir)
    datadir = rootdir + 'TE_process/data/'
    from TE import TE
    te = TE()
    return te, datadir


def test_pca():

    te, datadir = TE_instantiate()
    faultnr = '01'
    faultnr = '16'
    ftrain = '/home/thomas/Dropbox/software/TE/Tennessee_Eastman/TE_process/data/d'+faultnr+'.dat'
    ftest = '/home/thomas/Dropbox/software/TE/Tennessee_Eastman/TE_process/data/d'+faultnr+'_te.dat'

    mask = [0, 1, 16]
    #te.visualize_vars(infile=ftrain, dropfigfile='/tmp/outfigtrain.svg', title='Training Data', mask=mask)

    #te.visualize_vars(infile=ftest, dropfigfile='/tmp/outfigtest.svg', title='Test Data'); quit()
    #te.plotscatter('/home/thomas/Dropbox/software/TE/Tennessee_Eastman/te/out/all.csv')
    
    Xtrain = te.datacsvreadTE(ftrain)
    Xtest = te.datacsvreadTE(ftest)

    # Test data 48h with 960 samples = 20 samples/h --- Fault after 8h = 160 samples
    fault_data = Xtest[160:,:]
    print('Xtest.shape=', Xtest.shape, 'fault_data.shape=', fault_data.shape); # quit()


    X = Xtrain
    pca = PCA(n_components=2)
    pca.fit(X)
    show_eig(X, pca)
    #Y = pca.transform(X)
    #print('Y=', Y, '\n')
    show_result(pca)

    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(X)                 
    show_eig(X, pca)
    #Y = pca.transform(X)
    #print('Y=', Y, '\n')
    show_result(pca)

    pca = PCA(n_components=1, svd_solver='arpack')
    pca.fit(X)
    show_eig(X, pca)
    #Y = pca.transform(X)
    #print('Y=', Y, '\n')
    show_result(pca)
    '''
    When n_components is set to 'mle' or a number between 0 and 1
    (with svd_solver == 'full') this number is estimated from input data.
    '''
    pca = PCA(n_components='mle', svd_solver='full')
    #print('Data matrix shape=', X.shape)
    pca.fit(X)                 
    #show_eig(X)
    #Y = pca.transform(X)
    #print('Y=', Y, '\n')
    show_result(pca)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    varpercentage = 0.95    # find number of PCs with at least this percentage of variance
    numPCminvar = 1 + np.argmax(cumvar>varpercentage)
    print('Number of PCs with >= %.2f%%' % (100*varpercentage), 'variance = ', numPCminvar,
            'of a total of', cumvar.shape[0], 'components')
    plt.plot(cumvar)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.show()



'''
Michael E. Tipping and Christopher M. Bishop
Mixtures of probabilistic principal component analysers, Neural Computation 11(2),
pp 443-482. MIT Press., 1999
This version: June 26, 2006
http://www.miketipping.com/papers.htm
mail@miketipping.com
'''
def PPCA(T, n_components='mle', svd_solver='full', rotate=False, outfile=sys.stdout):
    # T is data matrix of the n d-dimensional observed samples
    n_samples, d = T.shape
    mu = np.mean(T, axis=0)
    '''
    #print('mu=', mu, 'var=', np.var(T, axis=0))
    # center the data and compute the sample covariance matrix
    T_centered = T - mu
    S = np.dot(T_centered.T, T_centered) / n_samples
    '''
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    pca.fit(T)     
    sigma2noise = pca.noise_variance_


    #print('pca.noise_variance_=', sigma2noise)            

    U = eigvec = pca.components_
    #print('U.shape=', U.shape, 'U=eigenvectors=\n', eigvec)
    U = U.T  # U is dxq matrix
    Lambda = eigval = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_
    q = numeigval = pca.n_components_
    print('$$$ PPCA: n_components=', n_components, 'svd_solver=', svd_solver,
            'Obtained number of eigenvalues=', q, 'Accumulated var=%.3f' % sum(var_ratio), '$$$', file=outfile)
    #print('Lambda=', Lambda, 'numeigval=', q, 'explained_variance_ratio_=',
    #       var_ratio, 'sum=%.3f' % sum(var_ratio), file=outfile)

    '''
    np.set_printoptions(threshold=sys.maxsize)
    print('U.shape=', U.shape, 'U=eigenvectors=\n', eigvec)
    print('phi\'*S*phi=', np.dot(eigvec.T, np.dot(S, eigvec)))
    print('q=', numeigval, 'of', d, 'eigvals=', eigval)    
    quit()
    '''
    
    sEye = sigma2noise*np.eye(q)
    L = np.sqrt(np.diag(Lambda) - sEye)
    W = np.dot(U, L)
    
    WWT = np.dot(W, W.T)
    
    WTW = np.dot(W.T, W)
    # Model covariance eq. (7)
    sEye_d =  sigma2noise*np.eye(d)
    C = WWT + sEye_d
    Cinv = LA.inv(C)

    '''
    print('L=\n', L, 'L cond=', LA.cond(L), '\nW.shape=', W.shape, 'W=\n', W, 'W cond=', LA.cond(W))
    print('WW\'=\n', WWT, 'WW\' cond=', LA.cond(WWT))
    print('W\'W=\n', WTW, 'WTW cond=', LA.cond(WTW))
    print('sEye_d\n', sEye_d, 'd=', q, 'sEye cond=', LA.cond(sEye_d))
    print('C=\n', C, 'shape=', C.shape, 'C cond=', LA.cond(C))
    CC = np.cov(T, rowvar=False)
    print('Covariance matrix of data=\n', CC, 'shape=', CC.shape, 'CC cond=', LA.cond(CC))
    show_result(pca)
    print('Cinv=\n', Cinv, 'Cinv cond=', LA.cond(Cinv))#, 'Cinv*C=', np.dot(Cinv, C))
    #quit()
    '''
    
    if rotate:
        Weigval, Weigvec = eigen(WTW)    # Eq. (62)
        R = Weigvec.T
        #print('R=\n', R)
        W = np.dot(W, R)
        #print('W rotated=\n', W)
    
    # return the three model parameters of the PPCA
    return mu, W, sigma2noise, C, Cinv, numeigval
    

def test_PPCA(Xtrain, Xtest):
    X = Xtrain
    print('\nX.shape=', X.shape)#, 'X=\n', X)
    n_samples, d, control_limit = PPCA_parameters(X)
    mu, W, sigma2noise, C, Cinv = PPCA(X)
    print('mu=', mu)
    print('\nW.shape=', W.shape, 'W=\n', W)
    print('\nC.shape=', C.shape, 'C=\n', C)
    print('\nCinv.shape=', Cinv.shape, 'Cinv=\n', Cinv)
    print('sigma2noise=', sigma2noise, 'control_limit=', control_limit)

    # Calculate the M2 for all x of the data matrix X Eq.(3)
    # data must be centered
    Xmean = np.mean(X, axis=0)
    X_centered = X - Xmean
    M2_X = np.sum( np.dot(X_centered, Cinv) * X_centered, axis=1) # [3], Eq. (4) ajusted to M2

    NormalConditionM2 = M2_X < control_limit
    numFaultsM2 = list(NormalConditionM2).count(False)
    print('M2_X=', M2_X)
    print('Normal Condition: M2 < control_limit=', NormalConditionM2)
    print('# Training data: Faults for M2 = ', numFaultsM2, 'of a total of ', n_samples, 'samples = %.2f' % (100.0*numFaultsM2/n_samples), '%')

    X_centered = Xtest - Xmean
    M2_X = np.sum( np.dot(X_centered, Cinv) * X_centered, axis=1) # [3], Eq. (4) ajusted to M2

    NormalConditionM2 = M2_X < control_limit
    numFaultsM2 = list(NormalConditionM2).count(False)
    print('M2_X=', M2_X)
    print('Normal Condition: M2 < control_limit=', NormalConditionM2)
    print('# Test data: Faults for M2 = ', numFaultsM2, 'of a total of ', n_samples, 'samples = %.2f' % (100.0*numFaultsM2/n_samples), '%')



####
# Simulation study from Eq. (28)
####
# ddof=1 ==> divide by (n-1) --- ddof=0 ==> divide by n
ddof_std = 0    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.std.html#numpy.std

def simulateX_pub8():
    numsimnormal = 1000
    #np.random.seed(0123)
    G = np.array([  # lines=sensors, columns=intrincsic states
        [-0.1670, -0.1352],
        [-0.5671, -0.3695],
        [-0.1608, -0.1019],
        [ 0.7574, -0.0563],
        [-0.2258,  0.9119]])
    numsensors, numstates = G.shape

    # Attention: Difference for Matlab and Python:
    # scale is standard deviation, not variance
    variance = 1.0
    stdev = scale = np.sqrt(variance)

    T = np.random.normal(loc=[0,0], scale=[scale, scale], size=(numsimnormal,numstates))
    #print('\nT.shape=', T.shape, 'T=\n', T)
    sensor_error_variance = 0.01
    sensor_error = np.random.normal(scale=np.sqrt(sensor_error_variance), size = (numsimnormal,numsensors))
    signal = np.dot(T, G.T) + sensor_error
    #print('\nsignal.shape=', signal.shape, 'signal=\n', signal)
    normalmean = signal.mean(axis=0)
    mormalstd = signal.std(axis=0, ddof=ddof_std)
    '''
    signal = (signal - normalmean) / mormalstd    # Standardize to zero mean, unit variance
    '''
    #print('Simulated signal: Mean=\n', np.mean(signal,axis=0), '\nStd=\n', np.std(signal,axis=0))
    normal = signal
    normal_labels = np.zeros([numsimnormal, numsensors], dtype=int)
    fault = None
    numsimfault = 1
    F = np.random.normal(loc=[0,0], scale=[scale, scale], size=(numsimfault,numstates))
    #print('\nF.shape=', F.shape, 'F=\n', F)
    faulty_sensor_error = np.random.normal(scale=np.sqrt(0.01), size = (numsimfault,numsensors))
    faulty_signal = np.dot(F, G.T) + faulty_sensor_error
    #print('faulty_signal G.t+error=', faulty_signal)

    faulty_signal = np.zeros([numsimfault,numsensors])
    #print('faulty_signal only zeros=', faulty_signal)
    faultmag4 = 1.8
    faulty_signal[0,3] += faultmag4
    #print('faulty_signal=', faulty_signal)
    #print('Mean=\n', normalmean)
    fault = normalmean - faulty_signal
    #print('\nfault.shape=', fault.shape, 'fault=\n', fault)
    '''
    quit()
    '''
    fault_labels = np.zeros([numsimfault, numsensors], dtype=int)
    return normal, normal_labels, fault, fault_labels 


def set_C_and_fault_from_pub():
    C = np.array([[0.060400, 0.154800, 0.043500, -0.124700, -0.098300],
        [0.154800, 0.496300, 0.136900, -0.427000, -0.240000],
        [0.043500, 0.136900, 0.049100, -0.122500, -0.063400],
        [-0.124700, -0.427000, -0.122500, 0.599700, -0.202000],
        [-0.098300, -0.240000, -0.063400, -0.202000, 0.926200]])
    print('\n\nFixed C from pub [1],Eq.(29): C=\n', C)
    fault = np.array([[-0.079, -0.59, -0.22, -1.78, -0.024]])  # single fault value from pub with magnitude 1.8 in fourth sensor
    #fault = np.array([[-0.079, -0.59, 1.49, -1.48, -0.024]])  # multi-sensor fault with magnitude 1.5 in third and fourth sensor
    #fault = np.array([[-0.079, -0.59, 0.49, -0.48, -0.024]])  # multi-sensor fault with magnitude 0.5 in third and fourth sensor
    print('\nFixed fault from pub=\n', fault)
    xmean = np.zeros(5)
    return C, fault, xmean

def bool2idxContributing(boolarray):
    return 1 + np.where(boolarray)[0]

# Attention: If the argument is not copied, the inversion is applied
# within the calling context
def bool2idxMissing(boolarray):
    return 1 + np.where(np.invert(np.copy(boolarray)))[0]


def test1():
    training_data, training_labels, fault_data, fault_labels = simulateX_pub8()
    X = training_data
    control_limit = PPCA_control_limit(X.shape[1])
    mu, W, sigma2noise, C, Cinv, numeigval = PPCA(X)

    #print('mu=', mu)
    #print('\nW.shape=', W.shape, 'W=\n', W)
    #print('\nC.shape=', C.shape, 'C=\n', C)
    #print('\nCinv.shape=', Cinv.shape, 'Cinv=\n', Cinv)
    #print('sigma2noise=', sigma2noise, 'control_limit=', control_limit)

    print('\n\nFrom data: C=\n', C)
    print('From data: fault=\n', fault_data)
    #print('Cinv=', Cinv)

    # Calculate the M2 for all x of the fault data matrix 

    print('ERROR. Does not work like this. Exit'); quit()
    M2_fault = np.sum( np.dot(fault_data, Cinv) * fault_data, axis=1)
    print('From data M2_fault=', M2_fault)

    '''
    C, fault, xmean = set_C_and_fault_from_pub()
    Cinv = LA.inv(C)
    #print('Cinv=', Cinv)
    faux = fault[0] #- mu
    #print('faux=', faux)
    print('From pub M2_fault=', np.dot(np.dot(faux, Cinv), faux))
    '''

    '''
    # Calculate the M2 for all x of the data matrix X
    X_centered = X - np.mean(X, axis=0)
    M2_X = np.sum( np.dot(X_centered, Cinv) * X_centered, axis=1)
    print('M2_X=', M2_X)

    NormalConditionM2 = M2_X < control_limit
    numFaultsM2 = list(NormalConditionM2).count(False)
    print('Normal Condition: M2 < control_limit=', NormalConditionM2)
    print('# Faults for M2 = ', numFaultsM2)
    '''

    # M2 contribution
    Cinvsqrt = sqrtm(Cinv)
    c_M2 = np.dot(X, Cinvsqrt)**2
    #print('c_M2=\n', c_M2)
    c_M2_fault = np.dot(fault_data, Cinvsqrt)**2
    print('\n M2 CONTRIBUTION:\nFrom data: c_M2_fault=\n', c_M2_fault)

    # T2 contribution
    D = calc_T2_D_mat(X)
    Dsqrt = sqrtm(D)
    c_T2_fault = np.dot(fault_data, Dsqrt)**2
    print('\n T2 CONTRIBUTION:\nFrom data: c_T2_fault=\n', c_T2_fault)


def M2_contribution_plot(C, train_data, test_data, contribution_plot=True, control_limit=None, barnames=None):
    """Consider only individual variables, ignoring all other variables
    """
    M = LA.inv(C)
    X = train_data
    n, numvar = X.shape

    #X = X[0].reshape((1,X.shape[1]))    # take only the first sample

    Msqrt = sqrtm(M)
    #print('M=\n', M, '\nMsqrt=\n', Msqrt, '\nMsqrt^2=', np.dot(Msqrt,Msqrt))
    #print('X.shape=', X.shape, 'X=\n', X)

    individual_contribs = np.dot(X, Msqrt)**2
    print('Training data: individual_contribs=', individual_contribs)
    M2_train = np.sum(individual_contribs, axis=1)
    print('Training data: M2=', M2_train)
    E_individual_contribs = np.mean(individual_contribs, axis=0)
    print('Training data: E_individual_contribs=', E_individual_contribs)

    X = test_data
    individual_contribs = np.dot(X, Msqrt)**2
    print('Test data: individual_contribs=', individual_contribs)
    M2_test = np.sum(individual_contribs, axis=1)
    print('Test data: M2=', M2_test)
    E_individual_contribs = np.mean(individual_contribs, axis=0)
    print('Test data: E_individual_contribs=', E_individual_contribs)

    if contribution_plot:
        # [1], Fig.2
        import matplotlib.pyplot as plt
        height = individual_contribs[0] # Pick only one sample to plot
        #bars = ('A', 'B', 'C', 'D', 'E')
        y_pos = np.arange(numvar)
        #y_pos = np.arange(len(height))
        plt.bar(y_pos, height, label=None)
        plt.xticks(y_pos, 1+y_pos, fontsize=7, rotation=90)
        if barnames != None:
            ax = plt.gca()
            for i, v in enumerate(height):
                ax.text(i, 0.3, barnames[i], fontsize=7, verticalalignment='bottom',
                        horizontalalignment='center', alpha=0.6, rotation=90, color='black')
        plt.xlabel('variable')
        plt.title('Individual Contribution Plot')
        plt.ylabel('contribution')
        if control_limit != None:
            plt.axhline(y=control_limit, color='r', linestyle='--', label='M2 control limit')
        plt.legend()
        #plt.show()

        timeevolution = plt.figure(2);
        plt.title('Temporal Evolution of Index')
        plt.xlabel('t')
        plt.ylabel('Index')
        if control_limit != None:
            plt.axhline(y=control_limit, color='r', linestyle='--', label='M2 control limit')
        plt.plot(M2_train, linewidth=0.5, color='green', label='Training data')
        plt.plot(M2_test, linewidth=0.5, color='blue', label='Test data')
        plt.legend()
        plt.show()


def test_M2_contribution_plot(contribution_plot=True):
    """Consider only individual variables, ignoring all other variables
    Do not consider conditional means and covariances
    """
    C, fault, xmean = set_C_and_fault_from_pub()
    control_limit = chi2.ppf(0.95, C.shape[0])  ###################
    M2_contribution_plot(C, fault, fault, contribution_plot, control_limit=control_limit)


def calc_conditional_mean_and_covar(xmean, C, contributing_vars):
    """Conditional mean and covariance matrix
    """
    numvar = xmean.shape[0]
    #print('xmean.shape=', xmean.shape, 'C.shape=', C.shape, 'contributing_vars.shape=', contributing_vars.shape )
    assert numvar==contributing_vars.shape[0] and C.shape[0]==C.shape[1] and numvar==C.shape[0], \
                 'Dimension problem in function extract_sub_region'

    #print('Calculating Conditional mean and covariance matrix ...')
    invcontributing_vars = np.invert(contributing_vars)
    #print('C=\n', C)
    Co = C[contributing_vars,:]
    #print('Co lines=\n', Co)
    Coo = Co[:,contributing_vars]
    #print('Coo=\n', Coo)
    Com = Co[:,invcontributing_vars]
    #print('Com=\n', Com)

    Cm = C[invcontributing_vars,:]
    #print('Cm lines=\n', Cm)
    Cmo = Cm[:,contributing_vars]
    #print('Cmo=\n', Cmo)
    Cmm = Cm[:,invcontributing_vars]
    #print('Cmm=\n', Cmm)

    # Reconstruct the permuted matrix for verification
    C_ordered = np.block([[Coo,Com],[Cmo,Cmm]])
    #print('C_ordered=\n',C_ordered)

    # The original C matrix in [1], Eq.5, and its inverse have to be
    # substituted by the reorganized version
    C = C_ordered
    Cinv = LA.inv(C)

    # Conditional mean
    Coo_inv = LA.inv(Coo)
    #print('Coo_inv=\n', Coo_inv)
    Cmo_x_Coo_inv = np.dot(Cmo, Coo_inv)

    # Conditional covariance matrix
    C_cond_mo = Cmm - np.dot(Cmo_x_Coo_inv, Com)    # Eq.(7)
    #print('C_cond_mo=\n', C_cond_mo)

    xomean = xmean[contributing_vars]
    invcontributing_vars = np.invert(contributing_vars)
    xmmean = xmean[invcontributing_vars]
    # The original mean vector x in [1], Eq.5 has to be substituted by the reorganized version
    xmean = np.concatenate((xomean,xmmean), axis=0)  # Eq.(5)

    #print('\ncontributing_vars=', contributing_vars, '\nxomean=', xomean, '\nxmmean=', xmmean)
    return xmean, xmmean, xomean, C, Cinv, Coo_inv, Cmo_x_Coo_inv, C_cond_mo
    

def calc_EM2(x, xmean, C, Cinv, contributing_vars, xmmean, xomean, Coo_inv, Cmo_x_Coo_inv, C_cond_mo, verbose=False):
    '''Split vector a and matrix C into subparts
    defined by the binary index vector 'contributing_vars'
    o=observed m=missing
    and calculate the expecxted M2 (Squared Mahalanobis distance statistic
    with the observed variables only
    '''
    if verbose:
        print('Calculating EM2 ...')
    y = (x-xmean)[contributing_vars]
    numvar = xmean.shape[0]
    invcontributing_vars = np.invert(contributing_vars)

    xo = x[contributing_vars]   # Eq.(4)
    xm = x[invcontributing_vars]    # Eq.(4)
    if verbose:
        with np.printoptions(precision=2, suppress=True):
            print('x=', x, '\ncontributing_vars=', contributing_vars, '\nxo=', xo, '\nxm=', xm)
    # https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution
    xmean_cond_mo = xmmean + np.dot(Cmo_x_Coo_inv, xo-xomean)   # Eq.(6)
    if verbose:
        with np.printoptions(precision=2, suppress=True):
            print('xmean_cond_mo=z_m=', xmean_cond_mo) 
          
    
    # Conditional mean of whole x
    xmean_cond_o = np.concatenate((xo, xmean_cond_mo), axis=0)  # Eq.(8)
    if verbose:
        with np.printoptions(precision=2, suppress=True):
            print('xmean_cond_o=z=', xmean_cond_o) 

    # Conditional covariance matrix of whole x
    sub_shape = C_cond_mo.shape
    #print('sub_shape=', sub_shape)
    lower = numvar - sub_shape[0]
    upper = lower + sub_shape[0]
    C_cond_o = np.zeros(C.shape)
    C_cond_o[lower:upper, lower:upper] = C_cond_mo  # Eq.(8)
    #print('C_cond_o=\n', C_cond_o) 


    # Re-calculate M2 statistic Eq. (9)
    diffvec = xmean_cond_o - xmean
    #print('diffvec=', diffvec) 
    M = np.outer(diffvec, diffvec) + C_cond_o
    #print('M=\n', M)
    M = np.dot(Cinv, M)
    #MM=np.eye(numvar)+np.dot(Cinv,C_cond_o) # DEBUG
    #print('\nM=\n', M, 'trace=', np.trace(M), '\nVerify: MM=\n', MM, 'trace=', np.trace(MM)); #quit()
    EM2 = np.trace(M) # https://math.stackexchange.com/questions/2228398/trace-trick-for-expectations-of-quadratic-forms
    #print('EM2 (Eq.(9))=', EM2)
    
    '''
    same result
    EM2 = np.trace(np.dot(Cinv, C_cond_o)) + np.dot(diffvec.T, np.dot(Cinv, diffvec))
    print('EM2 (Eq.(16))=', EM2)
    '''

    # Verification [1], after Eq. 16  tr(Cinv * C_cond_o) = d
    #print('Verification missing vars =', xm.shape[0], '=', np.trace(np.dot(Cinv, C_cond_o)))

    # Chen [2], Eq. (6)
    '''
    mu = xmean
    z = xmean_cond_o
    z_centered = z-mu
    Q = C_cond_o
    EM2 = np.trace(np.dot(Cinv, (np.outer(z_centered,z_centered)+Q )))
    print('EM2 (Chin Eq.(6))=', EM2)
    '''

    '''
    # simplified EM2, Eq. (17)  does not show the same value ==> have to check much better
    print('Calculating EM_2 in subspace of dim=', y.shape[0])
    # d can be omitted, since constant for fixed set of missing variables
    _d = np.trace(np.dot(Cinv, C_cond_o))
    #print('_d (following Eq. (16) of paper should be integer)=',_d, '\nNumber of missing vars=', d)
    EM2_simpl = np.dot(y.T, np.dot(Coo_inv, y)) # + _d
    print('EM2 simpl (Eq.(17))=', EM2_simpl, '_d=', _d, 'EM2_simpl+d=', EM2_simpl+_d )
    '''
    return EM2


def PPCA_parameters_OLD(X, beta=0.95):

    # https://en.wikipedia.org/wiki/Mahalanobis_distance
    # For a normal distribution in any number of dimensions, the probability density
    # of an observation is uniquely determined by the Mahalanobis distance d.
    # Specifically, d^2 is chi-squared distributed.
    def calc_control_limit(df, beta):
        return chi2.ppf(beta, df)

    n_samples, d = X.shape
    control_limit = calc_control_limit(d, beta)
    #control_limit_subspace = np.zeros(d)
    #for r in range(d):
    #    control_limit_subspace[r] = calc_control_limit(r, beta)
    return n_samples, d, control_limit

def PPCA_control_limit(df, beta=0.95):

    # https://en.wikipedia.org/wiki/Mahalanobis_distance
    # For a normal distribution in any number of dimensions, the probability density
    # of an observation is uniquely determined by the Mahalanobis distance d.
    # Specifically, d^2 is chi-squared distributed.
    control_limit = chi2.ppf(beta, df)
    return control_limit

#kkk
def monotonicity_test_4_b_and_b():
    from faultdetect import simulateX_pub1
    print('DATA: Qin paper')
    # Generate training and test data by simulation

    faultdirections = 'Random'
    # if you want only one variable, duplicate it in the list, e.g. (1,1)
    faultdirections = (2,6) # (2,6) # also relativly good # yyy
    num_fault_directions = len(faultdirections)
    training_data, training_labels, fault_data, fault_labels = simulateX_pub1(
			faultdirections, no_graph=True, dropfigdir=None)
    n, d = training_data.shape
    control_limit = PPCA_control_limit(d)
    n_components = 3        # Manually set number of principal components
    xmean, W, sigma2noise, C, Cinv, numeigval = PPCA(training_data,
			n_components=n_components, svd_solver='full', outfile=None)

    x = np.array([0.83, 1.86, 1.66,  1.70, -0.98, 3.12]) # Eq. 23 Franklin paper
    x = fault_data[1]
    

    print('=================\n\tBenchmark: Qin paper\n==================\n')
    print('faultdirections=',faultdirections,'num_principal_components=',n_components)
    all_vars = np.ones(d, dtype=bool)
    contributing_vars = np.array([1, 0, 1, 1, 1, 0], dtype=bool)
    print('contributing_vars=', contributing_vars)
    invcontributing_vars = np.invert(contributing_vars)
    unique, counts = np.unique(contributing_vars, return_counts=True)
    #print('unique.shape=', unique.shape, 'unique=', unique, 'counts=', counts)
    if unique.shape[0] == 1:
        if unique[0] == False:
            print('None of the ', d, 'variables considered')
            num_missing = d
            num_observed = 0
        else:
            print('All of the ', d, 'variables considered')
            num_missing = 0
            num_observed = d
    else:
        # number of missing variables
        num_missing = counts[unique==False][0]
        num_observed = contributing_vars.shape[0] - num_missing
        print(num_observed, 'of the ', d, 'variables considered')
    print('contributing_vars=', contributing_vars, 'unique=', unique, 'counts=', counts, 'number of missing=',
            num_missing, 'of total of ', contributing_vars.shape[0])

    xmean, xmmean, xomean, C, Cinv, Coo_inv, Cmo_x_Coo_inv, C_cond_mo = calc_conditional_mean_and_covar(
                               xmean, C, contributing_vars)

    M2 = np.dot(np.dot(x[0], Cinv), x[0])   # [1], Eq. (3)
    x = x.reshape((1,x.shape[0]))
    #EM2 = calc_EM2(x, xmean, C, Cinv, contributing_vars, xmmean, xomean,
    #                        Coo_inv, Cmo_x_Coo_inv, C_cond_mo, verbose=True)

    #M2 = calc_EM2_all_samples(fault_data, xmean, C, all_vars)
    # [1], Eq. (9)
    EM2 =  calc_EM2_all_samples(x, xmean, C, contributing_vars, verbose=True)
    print('M2=', M2, 'EM2=', EM2, 'Control limit=', control_limit)




def test2():

    training_data, training_labels, fault_data, fault_labels = simulateX_pub8()
    X = training_data
    d = X.shape[1]
    control_limit = PPCA_control_limit(d)
    #mu, W, sigma2noise, C, Cinv, numeigval = PPCA(X)

    C, fault, xmean = set_C_and_fault_from_pub()
    Cinv = LA.inv(C)
    #print('Cinv=', Cinv)
    faux = fault[0] #- mu
    #print('faux=', faux)
    M2 = np.dot(np.dot(faux, Cinv), faux)   # [1], Eq. (3)
    print('\nCONTRIBUTION OF ALL VARIABLES\n')
    print('From pub M2_fault=', M2, '> Control limit=', control_limit)


    contributing_vars = np.array([0, 0, 0, 0, 0], dtype=bool)   # EM2= 4.9999999999999964 = 5 trace of 5x5 unit matrix
    contributing_vars = np.array([1, 1, 1, 1, 1], dtype=bool)   # EM2= 242.9632101315627

    # Table 1: One variable is missing
    print('\nCONTRIBUTION WITH ONE VARIABLE MISSING\n')
    contributing_vars = np.array([0, 1, 1, 1, 1], dtype=bool)   # EM2= 240.40657435164354    {x_1}
    contributing_vars = np.array([1, 0, 1, 1, 1], dtype=bool)   # EM2=  66.81829965873605    {x_2}
    contributing_vars = np.array([1, 1, 0, 1, 1], dtype=bool)   # EM2= 232.49513191209962    {x_3}
    contributing_vars = np.array([1, 1, 1, 0, 1], dtype=bool)   # EM2=   2.9855108777387076  {x_4}
    contributing_vars = np.array([1, 1, 1, 1, 0], dtype=bool)   # EM2=  28.511918748335034   {x_5}

    # Table 2: Two variables are missing
    print('\nCONTRIBUTION WITH TWO VARIABLES MISSING\n')
    contributing_vars = np.array([1, 1, 0, 0, 1], dtype=bool)   # EM2=  3.626468640470213 mag 1.5, 3.626468640470213 mag 0.5   {x_3,x_4}
    contributing_vars = np.array([1, 0, 1, 1, 0], dtype=bool)   # EM2= 141.87733585897803 mag 1.5, 19.40041793137313 mag 0.5   {x_2,x_5}

    print('contributing_vars=', contributing_vars)
    invcontributing_vars = np.invert(contributing_vars)
    unique, counts = np.unique(contributing_vars, return_counts=True)
    #print('unique.shape=', unique.shape, 'unique=', unique, 'counts=', counts)
    if unique.shape[0] == 1:
        if unique[0] == False:
            print('None of the ', d, 'variables considered')
            num_missing = d
            num_observed = 0
        else:
            print('All of the ', d, 'variables considered')
            num_missing = 0
            num_observed = d
    else:
        # number of missing variables
        num_missing = counts[unique==False][0]
        num_observed = contributing_vars.shape[0] - num_missing
        print(num_observed, 'of the ', d, 'variables considered')
    #print('contributing_vars=', contributing_vars, 'unique=', unique, 'counts=', counts, 'number of missing=',
    #        num_missing, 'of total of ', contributing_vars.shape[0])

    xmean, xmmean, xomean, C, Cinv, Coo_inv, Cmo_x_Coo_inv, C_cond_mo = calc_conditional_mean_and_covar(xmean, C, contributing_vars)

    # [1], Eq. (9)
    EM2 = calc_EM2(fault[0], xmean, C, Cinv, contributing_vars, xmmean, xomean,
                            Coo_inv, Cmo_x_Coo_inv, C_cond_mo)
    print('EM2=', EM2, 'Control limit=', control_limit)


def calc_M2_one_sample(x, xmean, C, contributing_vars):
    all_contributing = np.unique(contributing_vars)[0] # First position is 'True' then

    if all_contributing:
        print('calc_M2_one_sample> ALL variables contributing. M2=(x-xm)Cinv(x-xm)')
        xc = x-xmean
        Cinv = LA.inv(C)
        M2 = np.dot(np.dot(xc,Cinv),xc)
        return M2

    xmean, xmmean, xomean, C, Cinv, Coo_inv, Cmo_x_Coo_inv, C_cond_mo = \
            calc_conditional_mean_and_covar(xmean, C, contributing_vars)
    M2 = calc_EM2(x, xmean, C, Cinv, contributing_vars, xmmean, xomean,
                            Coo_inv, Cmo_x_Coo_inv, C_cond_mo)
    return M2


def calc_EM2_all_samples(test_data, xmean, C, contributing_vars, verbose=False):
    all_contributing = np.unique(contributing_vars)[0] # First position is 'True' then
    ntest = test_data.shape[0]
    #print('calc_EM2_all_samples> all_contributing=', all_contributing)
    if verbose:
        print('@@@@@@ calc_EM2_all_samples># samples in win=', ntest,
             'contributing=', bool2idxContributing(contributing_vars),
	    	'missing=',  bool2idxMissing(contributing_vars))
    #quit()
    #if all_contributing:
    #    Cinv = LA.inv(C)

    xmean, xmmean, xomean, C, Cinv, Coo_inv, Cmo_x_Coo_inv, C_cond_mo = \
            calc_conditional_mean_and_covar(xmean, C, contributing_vars)
    if verbose:
        with np.printoptions(precision=2, suppress=True):
            print('xmean=', xmean, 'xmmean=', xmmean, 'xomean=', xomean, '\nC=\n', C,
				'\nCinv=\n', Cinv, '\nCoo_inv=\n', Coo_inv, '\nCmo_x_Coo_inv=\n', Cmo_x_Coo_inv,
				'\nC_cond_mo=Q_m=\n', C_cond_mo )

    #print('ntest=', ntest) ; quit()
    M2 = np.zeros(ntest)
    for i, x in enumerate(test_data):
        #print('x[',i,']=', x)
        if all_contributing:
            xc = x-xmean
            M2[i] = np.dot(np.dot(xc,Cinv),xc)
        else:
            M2[i] = calc_EM2(x, xmean, C, Cinv, contributing_vars,
                xmmean, xomean, Coo_inv, Cmo_x_Coo_inv, C_cond_mo, verbose=verbose)
        #print('M2[%4d]=' % i, M2[i], 'all contributing=', all_contributing)
        #print('\nC=', C, '\nCinv=', Cinv, 'M2[%4d]=' % i, M2[i] )
    EM2 = np.mean(M2)
    #print('--- n test=', ntest, 'EM2=', EM2)    ; quit()
    return EM2


def select(training_data, fault_data, k_forward=3, k_backward=2):

    '''#-------------------------------------
    C, fault, xmean = set_C_and_fault_from_pub()
    Cinv = LA.inv(C)
    #print('Cinv=', Cinv)
    training_data = np.zeros(1, C.shape[0])
    test_sample = fault[0] #- mu
    '''#-------------------------------------
    
    n_samples, d = training_data.shape
    control_limit = PPCA_control_limit(d)
    xmean, W, sigma2noise, C, Cinv, numeigval = PPCA(training_data)
    
    print('n_samples=', n_samples, 'd=', d, 'control_limit=', control_limit)
    test_sample = fault_data[0]



    # SBS to get those variables that contibuted most
    # inicially all variables are selected
    contributing_vars = np.ones(d, dtype=bool)
    # calc statistic for all variables
    #M2 = calc_M2_one_sample(test_sample, xmean, C, contributing_vars)
    #print('M2 for all variables =', M2)
    M2 = calc_EM2_all_samples(fault_data, xmean, C, contributing_vars)
    print('M2 for all variables =', M2); #quit()
    if M2 > control_limit:
        print('\nF A U L T  D E T E C T E D ', M2, '>', control_limit, '===========\n\n')

    k = 0
    contributing = contributing_vars
    print('\n--- F O R W A R D ---')
    num_contributing = d
    done = False
    while not done:
        #control_limit = chi2.ppf(0.95, num_contributing-1)  ###################
        #print('contributing=', contributing, '=', contributing*1, 'k=', k)
        print('missing=', bool2idxMissing(contributing))
        candidates = np.copy(contributing)
        score = np.zeros(d)
        minScorePos = 0
        for i in range(d):
            if contributing[i]:
                candidates[i] = False
                #print('#contributing=', num_contributing, 'Cand#=', i+1, 'contributing=',
                #        contributing, 'Candidates=', candidates) 
                ###score[i] = calc_M2_one_sample(test_sample, xmean, C, candidates)
                score[i] = calc_EM2_all_samples(fault_data, xmean, C, candidates)
                #print('candidates=', candidates, 'minScorePos=', minScorePos, 'score=', score,
                #        '\n\tscore[', i+1, ']=', score[i], 'score[minScorePos]=', score[minScorePos])
                if score[i] < score[minScorePos]:
                    print('Cand#', i+1, 'scored with', score[i], 'lower than current min=', score[minScorePos])
                    minScorePos = i
                #print('minScorePos=', minScorePos+1, 'score=', score, 'control_limit=', control_limit)
                candidates[i] = True

        print('\n---------------------------\n')
        k += 1
        if score[minScorePos] < control_limit:
            print(' < < < < < < Score=', score[minScorePos], 'is below control limit=', control_limit)
        done = score[minScorePos] < control_limit and k == k_forward
        #if not done:
        #    contributing[minScorePos] = False
        contributing[minScorePos] = False
        num_contributing -= 1
        #print('num_contributing=', num_contributing, 'score=', score, 'control_limit=', control_limit)
    #print('After forward: final contributing=', contributing, 'with score=', score[minScorePos])
    print('After forward: final missing=', bool2idxMissing(contributing), 'with score=', score[minScorePos])


    print('\n---B A C K W A R D ---')
    done = False
    while not done:
        #print('contributing=', contributing, '=', contributing*1, 'k=', k)
        print('missing=', bool2idxMissing(contributing))
        candidates = np.copy(contributing)
        score = np.zeros(d)
        minScore = np.inf
        for i in range(d):
            if not contributing[i]:
                candidates[i] = True
                #score[i] = calc_M2_one_sample(test_sample, xmean, C, candidates)
                score[i] = calc_EM2_all_samples(fault_data, xmean, C, candidates)

                #print('\n#contributing=', num_contributing, 'Cand pos=', i,
                #        'score=', score[i], 'contributing=', contributing, '=', contributing*1)

                print('missing=', bool2idxMissing(contributing))
                #print('candidates=', candidates, 'minScorePos=', minScorePos, 'score=', score,
                #        '\n\tscore[', i+1, ']=', score[i], 'score[minScorePos]=', score[minScorePos])
                if score[i] < minScore:
                    print('---> Cand at pos', i+1, 'scored with', score[i], 'lower than current min=', minScore)
                    minScorePos = i
                    minScore = score[i]
                #print('minScorePos=', minScorePos+1, 'score=', score, 'control_limit=', control_limit)
                candidates[i] = False
            #sleep(1.0)

        print('Including cand at pos', minScorePos+1, 'with min score=', minScore, 'as contributing!!!')
        print('\n---------------------------\n')
        k -= 1
        if score[minScorePos] < control_limit:
            print(' < < < < < < Score=', score[minScorePos], 'is below control limit=', control_limit)
        done = score[minScorePos] < control_limit and k == k_backward
        #if not done:
        #    contributing[minScorePos] = False
        contributing[minScorePos] = True
        num_contributing += 1
        #print('num_contributing=', num_contributing, 'score=', score)
    #print('After backward: final contributing=', contributing, '=', contributing*1, 'with score=', minScore)
    print('After forward: final missing=', bool2idxMissing(contributing), 'with score=', minScore)
    return contributing


def test_pub():
    #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #X = testiris()
    training_data, training_labels, fault_data, fault_labels = simulateX_pub8()
    X = training_data
    contributing = select(training_data, fault_data, k_forward=4, k_backward=2)

def testTE():
    print('Executing testTE() ....')
    #plotscatter(TE_featname())

    te, datadir = TE_instantiate()
    faultnr = '01'
    faultnr = '16'
    ftrain = '/home/thomas/Dropbox/software/TE/Tennessee_Eastman/TE_process/data/d'+faultnr+'.dat'
    ftest = '/home/thomas/Dropbox/software/TE/Tennessee_Eastman/TE_process/data/d'+faultnr+'_te.dat'

    mask = [0, 1, 16]
    #te.visualize_vars(infile=ftrain, dropfigfile='/tmp/outfigtrain.svg', title='Training Data', mask=mask)

    #te.visualize_vars(infile=ftest, dropfigfile='/tmp/outfigtest.svg', title='Test Data'); quit()
    #te.plotscatter('/home/thomas/Dropbox/software/TE/Tennessee_Eastman/te/out/all.csv')
    
    te.Xtrain = te.datacsvreadTE(ftrain)
    te.Xtest = te.datacsvreadTE(ftest)
    #te.standardize()

    # Test data 48h with 960 samples = 20 samples/h --- Fault after 8h = 160 samples
    fault_data = te.Xtest[160:,:]
    print('te.Xtest.shape=', te.Xtest.shape, 'fault_data.shape=', fault_data.shape); # quit()

    #test_PPCA(te.Xtrain, te.Xtest)
    #return

    X = te.Xtrain
    control_limit = PPCA_control_limit(X.shape[1])
    mu, W, sigma2noise, C, Cinv, numeigval = PPCA(X)

    # Calculate the M2 for all x of the data matrix X Eq.(3)
    # data must be centered
    X_train_centered = te.Xtrain - np.mean(te.Xtrain, axis=0)
    X_test_centered = te.Xtest - np.mean(te.Xtrain, axis=0)

    M2_contribution_plot(C, X_train_centered, X_test_centered, contribution_plot=True,
            control_limit=control_limit, barnames=te.featname)
    
    contributing = select(te.Xtrain, fault_data, k_forward=6, k_backward=2)


def testT2():
    print('Executing testT2() ....')

    te, datadir = TE_instantiate()
    faultnr = '01'
    faultnr = '16'
    ftrain = '/home/thomas/Dropbox/software/TE/Tennessee_Eastman/TE_process/data/d'+faultnr+'.dat'
    ftest = '/home/thomas/Dropbox/software/TE/Tennessee_Eastman/TE_process/data/d'+faultnr+'_te.dat'

    te.Xtrain = te.datacsvreadTE(ftrain)
    te.Xtest = te.datacsvreadTE(ftest)

    # Test data 48h with 960 samples = 20 samples/h --- Fault after 8h = 160 samples
    fault_data = te.Xtest[160:,:]
    print('te.Xtest.shape=', te.Xtest.shape, 'fault_data.shape=', fault_data.shape); # quit()

    X = te.Xtrain
    pca = PCA(n_components=0.95)    # if 0<n_com<1, consider the aculumated variance
    pca.fit(X)
    show_result(pca)
    l = pca.n_components_
    P = pca.components_
    L = pca.singular_values_
    LInv = np.diag(1.0/L)

    # T2 statistic
    print('Diagonal matrix of the inverse of the first ', l, 'eigenvalues L=\n', LInv)
    D = np.dot(np.dot(P.T, LInv), P)
    print('T2 matrix=\n', D, 'shape=', D.shape)

    XC = X - pca.mean_
    T2 = np.sum(np.dot(XC,D)*XC, axis=1)
    print('T2=\n', T2, 'shape=', T2.shape)


if __name__ == '__main__':
    #test_pca(); quit()
    #test1(); quit()
    #test_M2_contribution_plot(); quit()
    #test2(); quit()
    #test_pub(); quit()
    #testTE(); quit()
    #testT2(); quit()
    monotonicity_test_4_b_and_b(); quit()
