import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import copy
import pickle
from datetime import datetime
from src.ellipse import plot_ellipse
from src.t2t import t2t
from src.t2s import t2s
from src.CSTR_plot import cm2inch, tex_setup, read_X, dataframe2sklearn


def join_figs(dir, flist):
    numfigs = len(flist)
    nrows = 2
    ncols = 2
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 4))
    row = col = 0
    for i in range(numfigs-1):
        fn = dir + flist[i] + '.pkl'
        print('Opening figure file %s' % fn)
        with open(fn, 'rb') as fid:
            ax = pickle.load(fid)
            plt.plot(ax=ax)
        axs[i] = ax

        #axs[i].append(ax)
        #fig.axes.append(ax)
        #fig.add_axes(ax)
        #axs[row, col] = ax
        #axs[i].plot(ax)
        col += 1
        if col == ncols:
            col = 0
            row += 1
    plt.show()


# https://stackoverflow.com/questions/49503869/attributeerror-while-trying-to-load-the-pickled-matplotlib-figure
def load_pickle_plot_save_new_3Dview(fn, azim=110, elev=-150):
    fig, ax = plt.gcf(), plt.gca()
    plt.close()
    print('load_pickle_plot_save_new_view> Loading file\n\t%s' % fn)
    with open(fn, 'rb') as fid:
        ax = pickle.load(fid)

    ax.view_init(elev=elev, azim=azim)
    plt.show()

    # Save the plot with custom view
    ext = '.pgf'
    newplotfn = fn[0:-4] + '_new_view' + ext
    print('Saving original file', fn, 'with modified view as\n\t', newplotfn) # ; input('...')
    plt.axis('tight')
    plt.savefig(newplotfn, bbox_inches='tight')

    ext = '.pdf'
    newplotfn = fn[0:-4] + 'new_view' + ext
    print('Saving original file', fn, 'with modified view as\n\t', newplotfn) #; input('...')
    plt.axis('tight')
    plt.savefig(newplotfn, bbox_inches='tight')


def savefig(dropfigfile, ax):
    if dropfigfile is None:
        return
    plt.tight_layout()
    plt.savefig(dropfigfile)
    print('Saving figure in ', dropfigfile)
    # dropfigfilepdf = dropfigfile[0:-4] + '.pdf'
    dropfigfilepgf = dropfigfile + '.pgf'
    plt.savefig(dropfigfilepgf)
    print('Saving figure in ', dropfigfilepgf)
    dropfigfilepdf = dropfigfile + '.pdf'
    plt.savefig(dropfigfilepdf)
    print('Saving figure in ', dropfigfilepdf)
    # Save the plot in a pickle serial object
    dt = datetime.now().strftime('%Y_%m_%d__%H.%M.%S.%f')  # avoid ":"
    # picklefile = dropfigfile[0:-4] + '.pkl'
    picklefile = dropfigfile + '.pkl'
    with open(picklefile, 'wb') as fid:
        pickle.dump(ax, fid)
    print('Saving plot in pickle file ', picklefile, '...')  # ; input('...')


def plot_stat(cstr, stat_type='t2', alfa=0.95, limvar=0.95,
              standardize=True, force_a=None, dropfigfile=None):
    """Plot control limit subspace.
    """
    datafn = cstr.datafn
    print('Opening input data file %s.' % datafn)
    df = read_X(datafn=datafn)
    X, y, featname = dataframe2sklearn(df)
    n, m = X.shape
    # print('X=', X, 'X.shape=', X.shape, '\ny=', y, 'y.shape=', y.shape)
    labels = y

    classname = np.unique(labels)
    ynum = LabelEncoder().fit_transform(y)
    # print('ynum=\n', ynum)
    classlabel, idx = np.unique(ynum, return_index=True)
    # print('labels=', labels, 'classlabel=', classlabel, 'idx=', idx)
    idxt, classlabel = zip(*sorted(zip(idx, classlabel)))

    # print('AFTER SORT: classlabel=', classlabel, 'idxt=', idxt)

    idxt, classname = zip(*sorted(zip(idx, classname)))
    # print('classname=', classname, 'idxt=', idxt)

    # Suppose that the first part of X contains the normal labels
    idx_normal = [j for j in range(len(labels)) if labels[j] == 'normal']
    # print('\nidx_normal=', idx_normal)

    def checkConsecutive(lst):
        return sorted(lst) == list(range(min(lst), max(lst)+1))

    assert checkConsecutive(idx_normal), 'Normal condition not contiguous'
    n = len(idx_normal)
    Xn = X[:n, :]

    if standardize:
        scaler = StandardScaler()
        scaler.fit(Xn)
        Xn = scaler.transform(Xn, copy=True)

    # print('Xn=', Xn, 'shape=', Xn.shape)

    # https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    assert stat_type in ('t2', 'q', 'c'), \
           ('Statistic ' + stat_type + ' not allowed!')
    M = t2t(Xn, alfa=alfa, limvar=limvar, force_a=force_a)
    m = M['m']
    a = M['a']
    print('Number of principal components = ', a, 'of', m,
          'possible, stat_type=', stat_type)

    title = 'Training normal'
    title = None
    Yn = t2s(Xn, M, stat_type, title=title)
    # z = Yn['z']
    # print('z normal=', Yn['z'])
    stat_str = Yn['stat_str']
    control_limit = Yn['thr']

    fig, ax = plt.subplots()
    usetex = True
    tex_setup(usetex=usetex)

    widthcm = 12.0  # Real sizes later in the LaTeX file
    heigthcm = 10.0
    fig.set_size_inches([cm2inch(widthcm), cm2inch(heigthcm)])
    # fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    linewidth = 0.5
    fontsize = 7

    #label = ('Fault detection statistic=' + stat_str +
             #' ('+str(a)+' of '+str(m)+' principal components, percentile='
             #+ '{:.2f}'.format(limvar) + ')')
    #label_thr = ('PC ' + str(a) + ' of ' + str(m) +
    #             ' - Threshold=' + '{:.3f}'.format(control_limit))

    #print('label=', label, 'label_thr=', label_thr)

    # plt.title('Temporal Evolution of ' + tit)
    plt.xlabel('$t$ [min]')
    plt.ylabel(stat_str)

    legs = []

    limit_label = 'Control limit=' + '{:.3f}'.format(control_limit)
    leg = ax.axhline(y=control_limit, color='r', linestyle='--',
                     label=limit_label, linewidth=linewidth)
    legs.append(leg)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=6)

    intervals = []
    start = 0
    numcond = len(classname)
    for cond in range(numcond):
        idx_cond = [j for j in range(len(labels))
                    if labels[j] == classname[cond]]
        num_cond = len(idx_cond)
        stop = start + num_cond - 1
        a = start
        b = stop + 1
        # print('cond=%10s' % classname[cond], 'num_cond=%4d' % num_cond,
              # 'start=%4d' % start, 'stop=%4d' % stop, 'a=%4d' % a, 'b=%4d' % b)
        intervals.append([a, b, default_colors[cond], classname[cond]])
        start = stop + 1
    # print('numsample=', len(y), 'intervals=', intervals)   ; input('...')

    numintval = len(intervals)
    for i in range(numintval):
        a, b, col, cond = intervals[i]
        # print('interval ', i, '=', intervals[i])
        numpts = b - a
        Xintval = X[a:b, :]
        x = np.linspace(a, b, numpts)
        #print('a=', a, 'b=', b,
              #'\nx=\n', x, ' numpts=', numpts,
              #'len(Xintval)=', len(Xintval))  # ; input('...')
        if standardize:
            Xintval = scaler.transform(Xintval, copy=True)
        Y = t2s(Xintval, M, stat_type)
        z = Y['z']
        # print('z intval=', z, 'len=', len(z))
        if str(cond) != 'normal':
            label = 'Condition: Fault ' + cond
        else:
            label = 'Condition: ' + cond
        leg, = ax.plot(x, z, color=col, label=label, linewidth=linewidth)
        legs.append(leg)

    faults = cstr.faults
    fault_labels = []
    for f in faults:
        id = str(f.id)
        if f.is_sensor_fault:
            id = 'Sensor fault ' + id
        else:
            id = 'Fault ' + id
        fault_labels.append(id)

    col_trigger = ('c', 'm', 'y')
    for i in range(1, numintval):
        a, b, col, cond = intervals[i]
        if fault_labels is not None:
            label_fault = fault_labels[i-1] + ' triggered'
        else:
            label_fault = None
        ax.axvline(x=a, linestyle='--', linewidth=linewidth,
                   label=label_fault, color=col_trigger[i])

    title = ('Fault detection statistic=' + stat_str +
             ' ('+str(M['a'])+' of '+str(M['m'])
             + ' principal components, percentile='
             + '{:.2f}'.format(limvar) + ')')
    ax.set_title(title, fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc='upper left')  # loc='best')

    savefig(dropfigfile, ax)
    plt.show()


def plot_2_D_statistics(X, stat_type='t2', alfa=0.95, limvar=0.95,
                        force_a=None, plot_subspace=True):
    """Plot control limit subspace.
       X must be centralized
    """

    X = copy.copy(StandardScaler(with_std=False).fit_transform(X))  # centralize data matrix
    # check is X is centralized
    assert stat_type == 't2' or stat_type == 'q',\
           ('Statistic ' + stat_type + ' not allowed!')
    assert np.linalg.norm(X.mean(axis=0)) < 1e-7, 'Data matrix must be centralized'

    # Angle and its cosine between two vectors
    def angle_between(u, v):
        cosangle = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)  # -> cosine of the angle
        angle = np.arccos(np.clip(cosangle, -1, 1))  # angle in radians
        return angle, cosangle

    n, m = X.shape  # n samples, m variables, data matrix dimension
    M = t2t(X, alfa=alfa, limvar=limvar, force_a=force_a)
    # m = M['m']
    a = M['a']
    print('Number of principal components = ', a, 'of', m,
          'possible, stat_type=', stat_type)

    no_2D = not ((stat_type == 't2' and a == 2)
                 or (stat_type == 'q' and m - a == 2))
    if no_2D:
        print(('Cannot plot 2-D: Statistic=' +
               '%s Data dimension=%d Number of principal components=%d' %
               (stat_type, m, a)))
        return

    Y = t2s(X, M, stat_type, title=None)
    z = Y['z']
    thr = Y['thr']

    normal = np.zeros(n)
    for i in range(n):
        normal[i] = z[i] <= thr

    F = Y['F']
    P = M['P']  # Loading vectors, dimension(m x a)
    Ptil = M['Ptil']  # Loading vectors of residuals, dimension(m x m-a)
    S = M['S']  # Eigenvalue matrix, dimension (m x m)
    s = M['s']  # Eigenvalue vector, dimension m
    # print('P[:, :a]=\n', P[:, :a])
    # print('S=\n', S, '\ns=', s) ; input('...')
    # print('np.diag(1.0/np.sqrt(S[:a]))=\n', np.diag(1.0/np.sqrt(np.diag(S[:a]))))
    if stat_type == 't2':
        M = np.dot(P[:, :a], np.diag(1.0/np.sqrt(s[:a])) )
        newaxes = np.dot(np.identity(m), P[:, :a])
        subspace_xaxis_label = 'Principal component 1 (scaled)'
        subspace_yaxis_label = 'Principal component 2 (scaled)'
    elif stat_type == 'q':
        M = Ptil    # SPE: M = Ptil
        newaxes = np.dot(np.identity(m), M)
        subspace_xaxis_label = 'Residual subspace axis 1 (scaled)'
        subspace_yaxis_label = 'Residual subspace axis 2 (scaled)'
    # print('M=\n', M, 'shape=', M.shape); input('...')

    Xproj = np.dot(X, M)
    # print('Xproj=\n', Xproj, 'shape=', Xproj.shape); input('...')

    linewidth = 0.5
    fontsize = 7
    if plot_subspace:
        fig, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    else:
        fig, ax = plt.gcf(), plt.gca()
        ax2 = (ax, )
    stat_str = Y['stat_str']
    # fig.clf()

    for ax in ax2:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize-1)
        # ax.axis('auto') # auto, equal # https://matplotlib.org/3.1.1/api/_as_gen/
        ax.axis('auto')

    title = ('Fault detection statistic='+stat_str
             +'\nUsing '+str(a)+' of '+str(m)+' principal components'
             +' $==>$'+' '+stat_str+' subspace has 2 dimensions'
             +'\nControl limit threshold='+'{:.3f}'.format(thr))
    plt.suptitle(title, fontsize=fontsize+1)

    if plot_subspace:
        ax = ax2[0]
        plot_kwargs = {'color':'g','linestyle':'-','linewidth':1,'alpha':0.2}
        fill_kwargs = {'color':'g','alpha':0.1}
        a = b = np.sqrt(thr)
        angle = 0.0
        plot_ellipse(semimaj=a, semimin=b, phi=angle, x_cent=0.0, y_cent=0.0,
                    theta_num=1000, ax=ax, plot_kwargs=plot_kwargs,
                    fill=True, fill_kwargs=fill_kwargs, data_out=False,
                    cov=None, mass_level=0.68)
        ax.set_xlabel(subspace_xaxis_label, fontsize=fontsize)
        ax.set_ylabel(subspace_yaxis_label, fontsize=fontsize)

        backwardincol = 'm'
        backwardoutcol = 'c'

        for i in range(n):
            edgecolors = backwardoutcol if not normal[i] else backwardincol
            ax.scatter(Xproj[i, 0], Xproj[i, 1], s=10,
                       facecolors='none', edgecolors=edgecolors, linewidths=0.5)

        ax = ax2[1]

    angle, cosangle = angle_between(newaxes[0], newaxes[1])
    sinangle = np.sin(angle)
    #print('angle between largest eigenvector and x-axis=', angle, '=', angle/np.pi, 'pi')
    RotMat = np.array([[cosangle, -sinangle], [sinangle, cosangle]])

    limit = np.sqrt(thr)

    a = 1.0/np.sqrt(S[0, 0])
    b = 1.0/np.sqrt(S[1, 1])
    plot_kwargs = {'color': 'r', 'linestyle': '-', 'linewidth': 1, 'alpha': 0.1}
    fill_kwargs = {'color': 'r', 'alpha': 0.1}
    plot_ellipse(semimaj=limit*a, semimin=limit*b, phi=-angle, x_cent=0.0, y_cent=0.0,
                    theta_num=1000, ax=ax, plot_kwargs=plot_kwargs,
                    fill=True, fill_kwargs=fill_kwargs, data_out=False,
                    cov=None, mass_level=0.68)

    #print('RotMat=', RotMat)
    ScaleMat = np.array([[a, 0.0], [0.0, b]])
    Xback = np.dot(np.dot(Xproj, ScaleMat), RotMat)
    #print('Xback=\n', Xback)

    backwardincol = 'g'
    backwardoutcol = 'r'
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.set_xlabel('Original variable space axis 1', fontsize=fontsize)
    ax.set_ylabel('Original variable space axis 2', fontsize=fontsize)

    # ax.axis('auto')
    for i in range(n):
        edgecolors = backwardoutcol if not normal[i] else backwardincol
        ax.scatter(Xback[i, 0], Xback[i, 1], s=10,
                   facecolors='none', edgecolors=edgecolors, linewidths=0.5)
    plt.show()
