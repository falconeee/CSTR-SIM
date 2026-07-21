from __future__ import print_function
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.base import BaseEstimator

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")


from probabilistic_PCA import PPCA, calc_parameters, calc_conditional_mean_and_covar, calc_EM2
# http://rasbt.github.io/mlxtend/
from my_sequential_feature_selector import SequentialFeatureSelector as SFS
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS



class MyRegressor(BaseEstimator):
    def __init__(self, numfeat):
        self.numfeat = numfeat
        self.contributing_vars = np.ones(self.numfeat, dtype=bool)
        print('contributing_vars=', self.contributing_vars)
        
    def _set_contribvars_from_indices(self):
        dim = len(self.indices)
        for i in range(dim):
            self.contributing_vars[self.indices[i]] = False
        #print('contributing_vars=', self.contributing_vars)
        
    def _calc_M2(self, x, xmean, C, Cinv, contributing_vars, xmmean, xomean, Coo_inv, Cmo_x_Coo_inv, C_cond_mo):
        # [1], Eq. (9)
        M2 = calc_EM2(x, xmean, C, Cinv, contributing_vars, xmmean, xomean,
                        Coo_inv, Cmo_x_Coo_inv, C_cond_mo)
        #print('M2=', M2, 'Control limit=', control_limit)
        return M2


    def combine(self, inputs):
        EM2 = np.mean(inputs)
        #print('>MyRegressor.combine: returning EM2=', EM2)
        return EM2

    def predict(self, X):
        #print('>MyRegressor.predict:')
        #print('X.shape to be predicted=\n',X.shape)
        #print('X to be predicted=\n',X)

        # X is the test set of the cross validation,
        # only the candidate features are present
        # X can be ignored. The important information is in the
        # self.X_test samples that has all features
        # calculate the test statistic with the canditate set
        # have to know which of the features are the candidates
        
        #print('indices=', self.indices)
        
        # form the missing variable vector from the indices
        self._set_contribvars_from_indices()
        
        #print('self.C=', self.C); quit()
        
        xmean, xmmean, xomean, C, Cinv, Coo_inv, Cmo_x_Coo_inv, C_cond_mo = \
            calc_conditional_mean_and_covar(self.xmean, self.C, self.contributing_vars)
        
        ntest = X.shape[0]
        M2 = np.zeros(ntest)
        X = self.X_test # override useless X
        for i, x in enumerate(X):
            #print('x[',i,']=',x)
            M2[i] = self._calc_M2(x, xmean, C, Cinv, self.contributing_vars, 
                               xmmean, xomean, Coo_inv, Cmo_x_Coo_inv, C_cond_mo)
        return self.combine(M2)

    def classify(self, inputs):
        print('>MyRegressor.classify:')
        return np.sign(self.predict(inputs))

    def fit(self, X, Y, **kwargs):
        #print('>MyRegressor.fit: >>>>>>>>>>>>>>>>>>>>>>>>');
        #print('X.shape to be fitted=\n',X.shape)
        #print('Y.shape to be fitted=\n',Y.shape)
        #print('X=\n',X)
        #print('Y=\n',Y)
        #print('kwargs=', kwargs)
        
        fit_params = kwargs['fit_params']
        self.control_limit = fit_params['control_limit']
        self.X_test = fit_params['X_test']
        self.xmean = fit_params['xmean']
        self.C = fit_params['C']
        
        self.indices = kwargs['indices']
        #print('self.C=', self.C); quit()

    def get_params(self, deep = False):
        return {'numfeat':self.numfeat}

    def set_params(**params):
        self.numfeat = params[0]


'''
scoring:
If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.
'''
from sklearn.metrics import make_scorer

def my_score_func(y, y_pred, arg1, arg2, arg3):
    #print('>my_score_func:')
    #print('y=', y)
    #print('y_pred=', y_pred)
    return y_pred


'''
dummyX = np.empty([n,numfeat])
xmean = np.empty([numfeat])
C = np.empty([numfeat,numfeat])    # holds all covariance matrices
Cinv = np.empty([numfeat,numfeat])    # holds all inverse covariance matrices

from sklearn.model_selection import KFold

n_splits = 3
dummyX = np.empty([n,numfeat])
xmean = np.empty([n_splits,numfeat])
C = np.empty([n_splits,numfeat,numfeat])    # holds all covariance matrices
Cinv = np.empty([n_splits,numfeat,numfeat])    # holds all inverse covariance matrices
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

k = 0
for train_index, test_index in kfold.split(dummyX):
    print('k=',k,'train_index=', train_index, ' ---- test_index', test_index)
    X_train, X_test = X[train_index], X[test_index]
    print('\nX_train.shape=', X_train.shape, 'X_test.shape=', X_test.shape)
    n_samples, d, control_limit = calc_parameters(X_train)
    xmean[k], W, sigma2noise, C[k], Cinv[k] = PPCA(X_train)
    print('xmean=', xmean[k])
    print('\nW.shape=', W.shape, 'W=\n', W)
    print('\nC.shape=', C[k].shape, 'C=\n', C[k])
    print('\nCinv.shape=', Cinv[k].shape, 'Cinv=\n', Cinv[k])
    print('sigma2noise=', sigma2noise, 'control_limit=', control_limit)
    
    
    sfs = SFS(myregressor, k_features=2, forward=True, 
           floating=False, verbose=2,
           scoring=my_scorer, cv=None, n_jobs=-1)
    fit_params = {'n':n,'numfeat':numfeat,'feature_names':feature_names,'xmean':xmean[k],'C':C[k],'Cinv':Cinv[k],'X_test':X_test}
    Ydummy = np.zeros((X_test.shape[0],0))
    sfs = sfs.fit(X_test, Ydummy, fit_params=fit_params)
    
    k += 1
'''


def select_contributing(X_train, X_test, feature_names=None):
    """Sequential Feature Selection to isolate the contribution variables
        in a fault detection problem

    Arguments
    ----------
    X_train : A n x d data matrix with n samples and d variables, representing
              the normal operation
    X_train : A m x d data matrix with m samples and d variables, representing
              the faulty operation. The objective is to detect the responsible
              variables
    Parameters
    ----------

    Attributes
    ----------

    Examples
    -----------
    For usage examples, please see

    """

    print("TRAIN:", X_train)
    print("TEST:", X_test)

    n_samples, d, control_limit = calc_parameters(X_train)
    xmean, W, sigma2noise, C, Cinv = PPCA(X_train)
    print('xmean=', xmean)
    print('\nW.shape=', W.shape, 'W=\n', W)
    print('\nC.shape=', C.shape, 'C=\n', C)
    print('\nCinv.shape=', Cinv.shape, 'Cinv=\n', Cinv)
    print('sigma2noise=', sigma2noise, 'control_limit=', control_limit)


    myregressor =  MyRegressor(numfeat=d)
    my_scorer = make_scorer(my_score_func, greater_is_better=False,
                            arg1=1, arg2=2, arg3=3)


    sfs = SFS(myregressor, k_features=2, forward=True, 
            floating=False, verbose=2,
            scoring=my_scorer, cv=None, n_jobs=-1)
    fit_params = {'n':n,'numfeat':numfeat,'feature_names':feature_names,
                        'control_limit':control_limit,
                        'xmean':xmean,'C':C,'Cinv':Cinv,'X_test':X_test}
    Ydummy = np.zeros((X_test.shape[0],0))
    sfs = sfs.fit(X_test, Ydummy, fit_params=fit_params)
    print('\n')

    #print('sfs=',sfs)
    print('sfs.k_feature_idx_=', sfs.k_feature_idx_)
    #print('sfs.k_feature_names_=',sfs.k_feature_names_)
    print('sfs.k_score_=',sfs.k_score_)
    print('sfs.subsets_=',sfs.subsets_)

#quit()

def sfs_info(sfs):
    print('sfs=',sfs)
    print('sfs.k_feature_idx_=', sfs.k_feature_idx_)
    #print('sfs.k_feature_names_=',sfs.k_feature_names_)
    print('sfs.k_score_=',sfs.k_score_)
    print('sfs.subsets_=',sfs.subsets_)



from sklearn.model_selection import train_test_split


def test_iris():
    iris = load_iris()
    X = iris.data
    X_train, X_test = train_test_split(X, shuffle=False, train_size=100,
                                   random_state=None)
    n, numfeat = X.shape
    y = iris.target
    feature_names = ('sepal length', 'sepal width', 'petal length', 'petal width')


def test_rand():
    np.random.seed(0123)
    numfeat = 10
    n = 100
    X = np.random.normal(scale=1.0, size = (n,numfeat))
    X_train, X_test = train_test_split(X, shuffle=True, train_size=0.5,
                                   random_state=None)
    nd = 0  # number of dependent variables
    Y = np.random.normal(scale=0.5, size = (n,nd))
    '''
    print('X=\n',X)
    print('Y=\n',Y,'\n\n\n\n')
    '''

def test_from_pub():
    #-------------------------------------
    from probabilistic_PCA import set_C_and_fault_from_pub
    from numpy import linalg as LA
    from scipy.stats import chi2


    C, fault, xmean = set_C_and_fault_from_pub()
    Cinv = LA.inv(C)
    d = C.shape[0]
    beta = 0.95
    control_limit = chi2.ppf(beta, d)
    Cinv = LA.inv(C)
    #print('Cinv=', Cinv)
    X_test = fault
    myregressor =  MyRegressor(numfeat=d)
    my_scorer = make_scorer(my_score_func, greater_is_better=False,
                            arg1=1, arg2=2, arg3=3)


    feature_names = tuple(['feat_'+str(i+1) for i in range(d)])

    sfs = SFS(myregressor, k_features=3, forward=True, 
            floating=False, verbose=0,
            scoring=my_scorer, cv=None, n_jobs=4)   #, n_jobs=-1) # -1=use all CPUs

    fit_params = {'n':None,'numfeat':d, 'control_limit':control_limit,
                        'xmean':xmean,'C':C,'Cinv':Cinv,'X_test':X_test}
    Ydummy = np.zeros((X_test.shape[0],0))
    sfs = sfs.fit(X_test, Ydummy, custom_feature_names=feature_names, fit_params=fit_params)

    sfs_info(sfs)
    #quit()

    sfs.set_param(reset=False, forward=False, floating=False, k_features=2)
    sfs = sfs.fit(X_test, Ydummy, custom_feature_names=feature_names, fit_params=fit_params)
    sfs_info(sfs)

#-------------------------------------

test_from_pub(); quit()

select_contributing(X_train, X_test)
