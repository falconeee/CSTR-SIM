import numpy as np
import pickle
import matplotlib.pyplot as plt

import os
import sys
sys.path.append('../')
#sys.path.append('../../')

from defines import defines
rootdir = defines['rootdir']

# https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#print('pyplot.default_colors=', default_colors)

def I(prompt='...'): input(prompt)


def path_setup(verbose=False):
    incdirs = defines['includedirs']
    for i in range(len(incdirs)):
        sys.path.append(incdirs[i])
    if verbose:
        print('sys.path=', sys.path)  # ; quit()


def myexit(comment=''):
    sys.tracebacklimit = 0
    print(comment, end='')
    raise Exception('Forced exit() ...')


def wait():
    print('...')
    input('PRESS ENTER TO CONTINUE.')
    print('...')


def origin_label(modulefname):
    # __file__ is the pathname of the file from which the module was loaded
    srcfnam = modulefname[modulefname.rfind('/')+1:]
    # print('srcfnam=', srcfnam)
    label_origin = '\\label{origin:' + srcfnam +'}\\ifpeeplabels\\vspace{0.5cm}\\fi'
    print('PUT INTO LaTeX, before \\label{xxx:...}:', label_origin, '\n\t')


def fig2file(modulefname, fname, dryrun=False):
    """Save figure to file."""
    if modulefname is not None:
        srcfnam = modulefname[modulefname.rfind('/')+1:]
        label_origin = '\\label{origin:' + srcfnam + ':' + fname + \
                       '}\\ifpeeplabels\\vspace{0.5cm}\\fi'
        print('PUT INTO LaTeX, before \\label{fig:...}:', label_origin, '\n\t')
        outfigfilename = rootdir + fname
    else:
        outfigfilename = fname
    if not dryrun:
        if len(outfigfilename) > 4 and outfigfilename[-4] != '.':
            print('File name of graphic without extension. Appending \'.pdf\'')
            outfigfilename += '.pdf' 
        print('Saving figure to ', outfigfilename)
        plt.savefig(outfigfilename, bbox_inches='tight')
        # If extension is PDF, generate externally EPS file
        #if fname.endswith('.pdf'):
            #print('Graphics file is PDF. Coverting also to EPS ...')
            #cmd = 'pdf2ps ' + outfigfilename + ' ' + outfigfilename[:-4] + '.eps'
            #print('Executing on operationg system level:\n', cmd, '...')
            #os.system(cmd)
            #print(' done.')
        if fname.endswith('.svg'):
            print('Graphics file is SVG. Coverting also to EPS ...')
            inkscape = 'inkscape  --without-gui --export-eps='
            inkscape = '/usr/local/bin/Inkscape-9c6d41e-x86_64.AppImage -D --export-filename='
            cmd = inkscape+outfigfilename[:-4] + '.eps ' + outfigfilename
            print('Executing on operationg system level:\n', cmd, '...')
            os.system(cmd)
            print(' done.')
    else:
        print('Dryrun: No output generated!')


def cm2inch(cm): return cm/2.54


def inch2cm(inch): return inch * 2.54


def rad2deg(rad): return rad*180.0/np.pi


def deg2rad(deg): return deg*np.pi/180.0


def print_array(x, prefix='', max_line_width=None, formatstr='%.3f',
                end='\n', show_shape=False):
    formatter={'float_kind':lambda x: formatstr % x}
    # print('BEFORE: type(x)=', type(x), 'x=', x)
    if type(x) == str or type(x) == list:
       x = np.array(x)
    # print('AFTER: type(x)=', type(x), 'x=', x)
    print(prefix, np.array2string(x, max_line_width=max_line_width,
                                  formatter=formatter))
    if show_shape:
        print('shape=', x.shape)
    '''
    if type(x)==list:
        x = np.array(x)
    print(prefix, np.array2string(x, # max_line_width=max_line_width,
           formatter={'float_kind':lambda x: formatstr % x}))
    '''

def ffmt(f, formatstr='{:.4f}'):
    return formatstr.format(f)


def boolstr(b): return 'TRUE' if b else 'FALSE'

def save_obj(obj, name):
    if not name.endswith('.pkl'):
        name = name + '.pkl'
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    if not name.endswith('.pkl'):
        name = name + '.pkl'
    with open(name, 'rb') as f:
        return pickle.load(f)


# https://en.wikipedia.org/wiki/Symmetric_matrix
# Cholesky decomposition: A has to be a real positive-definite symmetric matrix
# https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
def is_pos_def(A):
    """Check if matrix is positive definite."""
    return np.all(np.linalg.eigvals(A) > 0)


def check_symmetric(A, tol=1e-8):
    """Check matrix symmetry."""
    return np.allclose(A, A.T, atol=tol)


def second_largest(numbers):
    ''' Second largest element of array '''
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None


def unique_rows(a):
    '''eliminates duplicate rows in a matrix'''
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def peep(vec, comment='', prec=4, end='\n'):
    norm = np.linalg.norm(vec)
    p = str(prec)
    formstr = '%s: (x,y)= (%+.'+p+'f,%+.'+p+'f); ||.||=%+.'+p+'f'
    # print('%s: (x,y)= (%.4f,%.4f); ||.||=%.4f' % (comment,vec[0],vec[1],norm))
    # print('prec=', prec, 'p=', p, 'formstr=', formstr)
    print(formstr % (comment, vec[0], vec[1], norm), end=end)


# Python numpy.linalg.eig does not sort the eigenvalues and eigenvectors
def eigen(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(eigenValues)
    idx = idx[::-1]  # Invert from ascending to descending
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return (eigenValues, eigenVectors)


# https://github.com/liyanage/python-modules/blob/master/running_stats.py
class RunningStats:
    """Online bookkeeping of mean and variance."""

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())



# LaTeX support: https://matplotlib.org/users/usetex.html

#plt.rc('text', usetex=True)
#plt.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
#
#params = {'legend.fontsize': 'small',
#          'figure.figsize': (15, 5),
#         'axes.labelsize': 'small',
#         'axes.titlesize':'small',
#         'xtick.labelsize':'small',
#         'ytick.labelsize':'small'}
# plt.rcParams.update(params)


def tex_setup(usetex=True):
    if not usetex:
        print('Not setting any Matplotlib parameters for LaTeX ...')
        return
    # import matplotlib.style
    # plt.style.use('classic')
    # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    # plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    # for Palatino and other serif fonts use:
    # plt.rc('font',**{'family':'serif','serif':['Palatino']})
    plt.rc('text', usetex=usetex)
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rc('font', **{'family': 'serif', 'serif': ['DejaVu Sans']})

    # Set the font size. Either an relative value of 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large' or an absolute font size, e.g., 12.
    # https://matplotlib.org/api/font_manager_api.html#matplotlib.font_manager.FontProperties.set_size
    texfigparams = {
        'axes.labelsize': 9,
        # 'axes.linewidth': 0.01,
        'axes.titlesize': 7,
        # 'figure.figsize': (8, 9),
        'font.size': 9,
        'legend.fontsize': 7,
        'legend.loc': 'lower center',
        # 'legend.loc': 'upper left',
        'lines.linewidth': 1,
        # 'savefig.dpi': 600,
        # 'savefig.bbox': 'tight',
        # 'ps.usedistiller': 'ghostscript',
        # 'ps.usedistiller': 'xpdf',
        'text.latex.preamble': r'\usepackage{amsmath}',
        'text.latex.preamble': r'\usepackage{amssymb}',
        'text.latex.preamble': r'\usepackage{bm}',
        #'text.latex.preamble': r'\usepackage{xcolor}',
        'pgf.preamble': r'\usepackage{amsmath}',
        'pgf.preamble': r'\usepackage{amssymb}',
        'pgf.preamble': r'\usepackage{bm}',
        #'pgf.preamble': r'\usepackage{xcolor}',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7
    }

    plt.rcParams.update(texfigparams)
    plt.rc('text', usetex=usetex)
    plt.rc('font', family='sans-serif')
    #print('Matplotlib: rcParams=\n', plt.rcParams, file=sys.stdout) ; quit()


def latex_tab_print_list(l, comment='', sep='&', formstr='%s'):
    #if type(l)==list:
    #    l = np.array(l)
    print(comment, end='')
    for i in range(len(l)):
        if formstr is not None:
            print(' %s ' % sep + formstr % (l[i]), end='')
        else:
            print(' %s ' % sep, l[i], end=' '+sep+' ')
            
    print(' \\\\')    

    
def latex_tab_print_matrix_numpy(M, comment='', sep='&', formstr='%6.2f', offset=0):
    #if type(M)==list:
        #M = np.array(M)
    is_vector = len(M.shape) == 1   # vector
    if is_vector:
        lin = 1
        col = M.shape[0]
    else:
        lin, col = M.shape
    print(comment, end='')
    for i in range(lin):
        for j in range(col):
            if is_vector:
                if formstr is not None:
                    print(' %s ' % sep + formstr % (M[j]+offset), end='')
                else:
                    print(' %s ' % sep, M[j], end=' '+sep+' ')
            else:
                if formstr is not None:
                    print(' %s ' % sep + formstr % (M[i][j]+offset), end='')
                else:
                    print(' %s ' % sep, M[i][j], end='')
        print(' \\\\')


def latex_tab_print_matrix(M, comment='', sep='&', formstr='%6.2f', offset=0):
    if type(M) != list:
        latex_tab_print_matrix_numpy(M, comment, sep, formstr, offset)
    else:
        lin = len(M)
        print(comment, end='')
        for i in range(lin):
            col = len(M[i])
            for j in range(col):
                if formstr is not None:
                    print(' %s ' % sep + formstr % (M[i][j]+offset), end='')
                else:
                    print(' %s ' % sep, M[i][j], end='')
            print('\t\\\\')


if __name__ == '__main__':
    fig2file(__file__, '/tmp/test', dryrun=False)
    #fig2file(None, '/tmp/lixo.svg', dryrun=False)
    fig2file(None, '/tmp/lixo.pdf', dryrun=False)
    fig2file(None, '/tmp/lixo1.eps', dryrun=False)
