import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import copy
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def cm2inch(cm): return cm/2.54


def label_set_cond(cond, labelstr):
    '''put label only when condition is satisfied'''
    if cond:
        return labelstr
    else:
        return None


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

    # Set the font size. Either an relative value of
    # 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large',
    # 'xx-large' or an absolute font size, e.g., 12.
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
        # 'text.latex.preamble': r'\usepackage{xcolor}',
        'pgf.preamble': r'\usepackage{amsmath}',
        'pgf.preamble': r'\usepackage{amssymb}',
        'pgf.preamble': r'\usepackage{bm}',
        # 'pgf.preamble': r'\usepackage{xcolor}',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7
    }

    plt.rcParams.update(texfigparams)
    plt.rc('text', usetex=usetex)
    plt.rc('font', family='sans-serif')
    # print('Matplotlib: rcParams=\n', plt.rcParams, file=sys.stdout) ; quit()


def filter_featname(featname, mask):
    if mask is None:
        return featname
    fn = [featname[i] for i in mask]
    return fn


def filter_vars(X, featname, mask):
    # print('filter_vars> mask=', mask, 'type=', type(mask))
    return (copy.copy(X[:, np.array(mask, dtype=int)]),
            filter_featname(featname, mask))


def read_X(datafn, sep=';'):
    """Read data matrix into a pandas dataframe."""
    df = pd.read_csv(datafn, sep=';')

    '''
    print(df)
    print('index: ', df.index)
    print('columns: ', df.columns)
    for col in df.columns:
        print('col: ', col, type(col), 'left stripped:', col.lstrip())
    classlabel = df.columns[-1].lstrip()
    print('Classlabel=', classlabel)
    print('info: ', df.info)
    print('head: ', df.head())
    print('df.head().iloc[-1]=', df.head().iloc[-1])
    print('shape: ', df.shape); input('...')
    raise Exception

    from tabulate import tabulate
    print(tabulate(df, headers='keys', tablefmt='psql'))
    '''
    return df


def dataframe2sklearn(df):
    """Convert a pandas dataframe into sklearn readable format X, y, Y."""
    featname = list(df.columns.values)
    featname = [f.lstrip() for f in featname]

    X = df.iloc[:, 0:-1].values
    # labels = df.iloc[:, -1]
    # classlabel = df.columns[-1].lstrip()
    # labels = df[classlabel].astype(str)

    lastcol = df.iloc[:, -1]
    labels = pd.Series(lastcol).values
    # print('\n', labels)


#    data = df.to_numpy()
#    X = data[:, 0:-1]
#    labels = data[:, -1]

    # print('data=\n', data, 'shape=', data.shape)
    # print('X=\n', X, 'shape=', X.shape)
    # print('featname=\n', featname)
    # print('labels=\n', labels, 'shape=', labels.shape)
    return X, labels, featname


def plot_signals(cstr, mask=None, title='CSTR Variables', dropdir=None, block=False):
    """Plot multichannel signal using Plotly."""
    
    datafn = cstr.datafn
    print(f'Opening input data file: {datafn}')
    
    # Carrega os dados gerados
    df_dataset = pd.read_csv(datafn, sep=";")
    
    # Seleciona apenas as colunas numéricas
    df_num = df_dataset.select_dtypes(include=['number'])
    colunas = df_num.columns
    
    # Aplica a máscara se houver (para plotar apenas variáveis específicas)
    if mask is not None:
        colunas = [colunas[i-1] for i in mask if (i-1) < len(colunas)]
        df_num = df_num[colunas]

    num_variaveis = len(colunas)
    
    # Cria os subplots empilhados
    fig = make_subplots(
        rows=num_variaveis, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.005
    )

    # Adiciona cada variável
    for i, col in enumerate(colunas):
        fig.add_trace(go.Scatter(
            x=df_num.index, 
            y=df_num[col], 
            mode='lines', 
            name=col
        ), row=i+1, col=1)
        
        fig.update_yaxes(title_text=col, row=i+1, col=1)

    # Layout final
    fig.update_layout(
        height=120 * num_variaveis, 
        title=title,
        hovermode='x unified',
        showlegend=False
    )

    fig.show()


def plot_condition(X, y, ynum, classlabel, classname, featname,
                   plot_time_axis=True,
                   time_offsets=None, dropfigfile=None,
                   title=None, block=True, azim=-45, elev=30):
    '''Given a set of patters with class label, plot in 2D.
    If the time axis option is true, plot the postion in time, following
    the order in the data matrix X (first pattern X[0] at t=0
    '''
    n, m = X.shape
    # print('CSTR.plot_condition>\nX=\n', X, 'shape=', X.shape,
    #      '\ny=\n', y, 'shape=', y.shape); input('...')
    assert m == 2 or m == 3, 'te.plot_condition> Data must be 2-D or 3-D'
    # if m == 3:
    #    plot_time_axis = False

    if plot_time_axis:
        print('CSTR.plot_condition> Generating 2-D plot with time evolution ...')
    elif m == 2:
        print('CSTR.plot_condition> Generating 2-D plot ...')
    else:
        print('CSTR.plot_condition> Generating 3-D plot ...')

    numclasses = len(classname)
    xlab = featname[0]
    ylab = featname[1]
    tex_setup(usetex=True)
    # fig, ax = plt.gcf(), plt.gca()
    # fig, ax = plt.subplots(); # Create a figure and a set of subplots
    cmap = plt.cm.tab20
    cmap = plt.cm.Paired
    cmap = plt.get_cmap('gnuplot')
    cmap = cmap.resampled(numclasses)
    colors = [cmap(i) for i in np.linspace(0, 1, numclasses)]
    colors = ('g', 'r', 'b', 'm', 'c')
    # https://matplotlib.org/stable/api/markers_api.html
    if m == 2:
        marker = 'x'
    else:
        marker = '.'
    # marker = '.' # 'x' '+' '.'
    markersize = 10
    linewidths = 0.5
    fontsize = 8
    plotargs = {'edgecolor': None, 'marker': marker,  # 'cmap': cmap,
                's': markersize, 'linewidths': linewidths}  # , 'fontsize': fontsize}
    if plot_time_axis:
        if title is not None:
            title = title + ' --- 2-D plot with time evolution'
        else:
            title = '2-D plot with time evolution'
        zlab = 't'
        ax = plt.figure().add_subplot(projection='3d', azim=azim, elev=elev)
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlab, fontsize=fontsize)
        # ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel(ylab, fontsize=fontsize)
        # ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel(zlab, fontsize=fontsize)
        # ax.w_zaxis.set_ticklabels([])
        for i in range(numclasses):
            condstr = str(classname[i])
            if condstr != 'normal':
                condstr = 'Fault ' + condstr
            idx = np.where(ynum == classlabel[i])
            numpts = len(idx[0])
            if time_offsets is not None:
                toffset = time_offsets[i]
            # print('i=', i, 'classlabel[i]=', classlabel[i], 'numpts=',
            #       numpts, 'toffset=', toffset) ; input('...')
            t = np.linspace(toffset, toffset+numpts-1, numpts)
            label = condstr + ': ' + str(len(idx[0])) + ' samples'
            # print('\nlabel=', label, 't=', t)
            ax.scatter(X[idx, 0], X[idx, 1], t, color=colors[i],
                       label=label, **plotargs)

            '''
            # Plot also a projetion on t=-500
            ax.scatter(X[idx, 0], X[idx, 1], -500, color=colors[i],
                        alpha=0.2, s=5,
                        label=label)
            '''

            # print('CSTR.plot_condition> i=', i, 'numclasses=', numclasses); input('...')
        # ax.set_zlim(bottom=0, top=toffset+numpts)
    elif m == 2:
        ax = plt.gca()
        # careful: iteration goes for the list with less elements
        for j, i, color in zip(range(numclasses), classlabel, colors):
            # print('y=\n', y, 'shape=', y.shape, 'yunique=', yunique, 'numclasses=', numclasses,
            #      'classname=', classname, 'j=', j, 'i=', i)
            idx = np.where(ynum == classlabel[i])
            # print('y=\n', y, 'numclasses=', numclasses, 'j=', j, 'i=', i, 'len(idx[0])=', len(idx[0]))
            label = classname[j] + ': ' + str(len(idx[0]))
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=label, **plotargs)
        plt.xlabel(xlab, fontsize=fontsize)
        plt.ylabel(ylab, fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
    else:
        zlab = featname[2]
        ax = plt.figure().add_subplot(projection='3d', azim=azim, elev=elev)
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlab, fontsize=fontsize)
        ax.set_ylabel(ylab, fontsize=fontsize)
        ax.set_zlabel(zlab, fontsize=fontsize)
        toffset = 0
        for i in range(numclasses):
            condstr = str(classname[i])
            if condstr != 'normal':
                condstr = 'Fault ' + condstr
            idx = np.where(ynum == classlabel[i])
            numpts = len(idx[0])
            t = np.linspace(toffset, toffset+numpts-1, numpts)
            if time_offsets:
                toffset += numpts
            label = condstr + ': ' + str(len(idx[0]))
            # print('y=\n', y, 'shape=', y.shape)
            # print('yunique=', yunique, 'numclasses=', numclasses,
            #      'classname=', classname, 'i=', i, 'numpts=', numpts,
            #      'label=', label) ; input('...')
            ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], color=colors[i],
                       label=label, **plotargs)

    ax.tick_params(axis='both', which='major', labelsize=fontsize-1)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-1)

    plt.legend(fontsize=fontsize, loc='best')
    plt.axis('tight')
    if dropfigfile is not None:
        plt.savefig(dropfigfile)
        print('Saving figure in ', dropfigfile)
        # dropfigfilepdf = dropfigfile[0:-4] + '.pdf'
        dropfigfilepgf = dropfigfile + '.pgf'
        plt.savefig(dropfigfilepgf)
        print('Saving figure in ', dropfigfilepgf)
        dropfigfilepdf = dropfigfile + '.pdf'
        plt.savefig(dropfigfilepdf)
        print('Saving figure in ', dropfigfilepdf)
        dropfigfileeps = dropfigfile + '.eps'
        plt.savefig(dropfigfileeps)
        print('Saving figure in ', dropfigfileeps)
        # Save the plot in a pickle serial object
        dt = datetime.now().strftime('%Y_%m_%d__%H.%M.%S.%f')  # avoid ":"
        # picklefile = dropfigfile[0:-4] + '.pkl'
        picklefile = dropfigfile + '.pkl'
        with open(picklefile, 'wb') as fid:
            pickle.dump(ax, fid)
        print('Saving plot in pickle file ', picklefile, '...')  # ; input('...')

    plt.show(block=block)


# Python numpy.linalg.eig does not sort the eigenvalues and eigenvectors
def eigen(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(eigenValues)
    idx = idx[::-1]  # Invert from ascending to descending
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return (eigenValues, eigenVectors)


def plot_PCA(cstr, X, y, plot_time_axis=True,
              dropfigfile=None, title=None, azim=-24, elev=7):
    print('Generating PCA plot...')
    labels = y
    ynum = LabelEncoder().fit_transform(y)

    meanX = np.mean(X, axis=0)
    Xcentered = X - meanX
    C = np.cov(X, rowvar=False)  # Observations (samples) are the rows
    print('Covariance Matrix of Dataset C=\n', C)

    Lambda, PHI = eigen(C)
    print('Eigenvectors (=columns) of Covariance Matrix of Dataset PHI=\n',
          PHI)
    print('PHI * PHI\'=\n', np.dot(PHI, PHI.T))
    print('Eigenvalues of Covariance Matrix of Dataset Lambda=\n', Lambda)
    print('C * PHI=\n', np.dot(C, PHI))
    print('\nLAMBDA * PHI=\n', np.dot(np.diag(Lambda), PHI))

    X = np.dot(Xcentered, PHI)
    X = X[:, 0:3]
    featname = ('PC 1', 'PC 2', 'PC 3')
    # print('ynum=\n', ynum)

    classlabel, idx = np.unique(ynum, return_index=True)
    # print('classlabel=', classlabel, 'idx=', idx)

    idxt, classlabel = zip(*sorted(zip(idx, classlabel)))
    time_offsets = idxt
    # print('classlabel=', classlabel, 'idxt=', idxt)

    classname = np.unique(labels)
    _, classname = zip(*sorted(zip(idx, classname)))

    plot_condition(X, y, ynum, classlabel, classname, featname,
                   plot_time_axis=plot_time_axis,
                   time_offsets=time_offsets, dropfigfile=dropfigfile,
                   title='Principal Components', block=True, azim=azim, elev=elev)



def plot_tSNE(X, y, n_components=3, plot_time_axis=False,
              dropfigfile=None, title=None, azim=-56, elev=52):
    print('Generating tSNE plot...')
    n, m = X.shape
    tsne = TSNE(n_components=n_components, learning_rate='auto',
                init='pca')
    X = tsne.fit_transform(X)
    ynum = LabelEncoder().fit_transform(y)
    classlabel, idx = np.unique(ynum, return_index=True)
    idxt, classlabel = zip(*sorted(zip(idx, classlabel)))
    time_offsets = idxt
    # print('classlabel=', classlabel, 'idxt=', idxt)

    classname = np.unique(y)
    # if n_components == 3:
    #    plot_time_axis = False

    title = ('t-distributed stochastic neighbor embedding (t-SNE):' +
             '\nMapping ' + str(m) + ' dimensions to ' +
             str(n_components) + ' dimensions' +
             '\n' + str(len(classname)) + ' different process conditions')
    if plot_time_axis:
        featname = ('t-SNE axis 1', 't-SNE axis 2', 'time')
    else:
        featname = ('t-SNE axis 1', 't-SNE axis 2', 't-SNE axis 3')
    plot_condition(X, y, ynum, classlabel, classname,
                   featname,
                   plot_time_axis=plot_time_axis,
                   time_offsets=time_offsets,
                   dropfigfile=dropfigfile,
                   title=title, azim=azim, elev=elev)


def plotscatter(cstr, feat1, feat2, feat3=None,
                standardize=False,
                title='CSTR: Condition in Feature Space'):
    """
    Gera um gráfico de dispersão (scatter plot) interativo 2D ou 3D das variáveis do CSTR.
    """
    datafn = cstr.datafn
    print(f'Opening input data file: {datafn}')
    
    # 1. Carrega os dados diretamente pelo Pandas
    df = pd.read_csv(datafn, sep=";")
    
    # Remove espaços em branco do nome das colunas (para evitar erros ao chamar a coluna CLASS)
    df.columns = df.columns.str.strip()
    
    # Garante que as features existem no DataFrame. Subtrai 1 se seus parâmetros (feat1, etc) usarem base 1.
    colunas_numericas = df.select_dtypes(include=['number']).columns
    var1 = colunas_numericas[feat1 - 1]
    var2 = colunas_numericas[feat2 - 1]
    
    colunas_filtro = [var1, var2]
    var3 = None
    if feat3 is not None:
        var3 = colunas_numericas[feat3 - 1]
        colunas_filtro.append(var3)

    # 2. Padronização (Opcional)
    if standardize:
        print('Standardizing data (Z-score)...')
        scaler = StandardScaler()
        # Aplica padronização apenas nas colunas numéricas selecionadas
        df[colunas_filtro] = scaler.fit_transform(df[colunas_filtro])
        
        # Atualiza os nomes das variáveis nos eixos para indicar a padronização
        var1_label = f"{var1} (Standardized)"
        var2_label = f"{var2} (Standardized)"
        var3_label = f"{var3} (Standardized)" if var3 else None
        
        df = df.rename(columns={var1: var1_label, var2: var2_label})
        if var3:
            df = df.rename(columns={var3: var3_label})
            
        var1, var2, var3 = var1_label, var2_label, var3_label

    # Adiciona a coluna de tempo/amostra para visualizar no hover
    df['Time_Index'] = df.index

    # 3. Geração do Gráfico Plotly
    # O parâmetro 'color' agrupa automaticamente os pontos pela classe de falha
    if feat3 is None:
        # Plot 2D
        fig = px.scatter(
            df, 
            x=var1, 
            y=var2, 
            color='CLASS', # Colore de acordo com a falha (normal, S2, 2+3, etc)
            hover_data=['Time_Index'], # Mostra a iteração do tempo no mouse
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set1 # Paleta de cores nítida
        )
    else:
        # Plot 3D
        fig = px.scatter_3d(
            df, 
            x=var1, 
            y=var2, 
            z=var3, 
            color='CLASS', 
            hover_data=['Time_Index'],
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        # Ajusta as proporções e remove fundo do gráfico 3D para ficar mais limpo
        fig.update_layout(
            scene=dict(
                xaxis_title=var1,
                yaxis_title=var2,
                zaxis_title=var3,
                aspectmode='cube'
            )
        )

    # Melhorias gerais no layout
    fig.update_layout(
        legend_title_text='Condição do Reator',
        height=700, # Gráfico maior para facilitar a visualização da dispersão
        hovermode='closest'
    )

    fig.show()


def train_test_pair_signal_plot(cstr, exp_train, exp_test, mask=None,
                                title=None, dropdir=None,
                                figext='pgf', block=True):

    print(('CSTR.train_test_pair_signal_plot>' +
           '\n\tPlotting the precalculated CSTR file ' +
           'training-test pair signal...'))

    # Load the data files of a pair of experiments
    id1 = exp_train['id']
    id1 = 'X_py' if id1 is None else id1
    datafn = cstr.datarootdir + id1 + '.csv'
    print('Opening input data file %s.' % datafn)
    df = read_X(datafn=datafn)
    Xtrain, ytrain, featname = dataframe2sklearn(df)

    id2 = exp_test['id']
    id2 = 'X_py' if id2 is None else id2
    datafn = cstr.datarootdir + id2 + '.csv'
    print('Opening input data file %s.' % datafn)
    df = read_X(datafn=datafn)
    Xtest, ytest, featname = dataframe2sklearn(df)

    NORMVAL = cstr.NORMVAL
    if mask is not None:
        Xtrain, _ = filter_vars(Xtrain, featname, mask)
        Xtest, featname = filter_vars(Xtest, featname, mask)
        NORMVAL = filter_featname(cstr.NORMVAL, mask)

    numfeat = Xtrain.shape[1]

    ntrain = Xtrain.shape[0]
    ntest = Xtest.shape[0]
    X = np.concatenate((Xtrain, Xtest), axis=0)
    XPlot = X
    numx = XPlot.shape[0]
    x = np.linspace(0, numx-1, numx)
    # print('ntrain=', ntrain, 'ntest=', ntest, 'numx=', numx)

    tex_setup(usetex=True)
    # https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # print('pyplot.default_colors=', default_colors)
    color = default_colors
    color_lim = default_colors[5:]
    # color = ('g', 'y', 'r', 'b', 'm', 'g', 'y', 'r', 'b', 'm')
    # print('color_lim=', color_lim)

    if numfeat > 1:
        fig, ax = plt.subplots(numfeat, 1)
    else:
        fig, ax1 = plt.gcf(), plt.gca()
        ax = [ax1, ]
    widthcm = 11  # Real sizes later in the LaTeX file
    heigthcm = 14
    nlegcol = 3
    fontsize = 7
    fig.set_size_inches([cm2inch(widthcm), cm2inch(heigthcm)])
    ax[numfeat-1].set_xlabel('$t$', fontsize=fontsize)
    # print('X.shape=', X.shape, 'X[:,0].shape=', X[:,0].shape,
    # 'X[:,1].shape=', X[:,1].shape)

    # vcolor = (0.8, 0.8, 0.8, 1)
    # vcolor = (0.3, 0.3, 0.3, 1)
    linewidth = 0.75
    fontsize = 7
    # plotargs = {'linewidth': linewidth, 'fontsize': fontsize}

    numx = XPlot.shape[0]
    usetex = True

    limlabels = []
    trainstart = 0
    trainstop = ntrain - 1
    teststop = ntrain - 1 + ntest - 1
    #  whitespace ignored: LaTeX issue ?
    # https://stackoverflow.com/questions/63459825/how-to-get-leading-whitespace-in-matplotlib-labels-legend-or-xlabel-etc-using
    limlabels.append((trainstart, '{:4d}'.format(trainstart) + ': train start'))
    limlabels.append((trainstop, str(trainstop) + ': train stop = test start'))
    limlabels.append((teststop, str(teststop) + ': test stop'))
    numlimlabels = len(limlabels)
    for s in range(numfeat):
        ylabel = featname[s]
        ax[s].set_ylabel(ylabel, rotation=0, labelpad=0,
                         verticalalignment='center',
                         horizontalalignment='right',
                         usetex=usetex, fontsize=8)
        ax[s].xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax[s].tick_params(axis='both', which='major', labelsize=fontsize-1)
        # ax[s].tick_params(axis='both', which='minor', labelsize=fontsize-1)

        for i in range(numlimlabels):
            t = limlabels[i][0]
            label = limlabels[i][1]
            label = label_set_cond(s == 0, label)
            ax[s].axvline(x=t, linestyle=':', color=color_lim[i],
                          linewidth=linewidth, label=label)

    faults = exp_train['faults']

    cond = 'normal'
    t = 0
    start = t
    i = 0
    for f in faults:
        ftriggered = f.DELAY
        if ftriggered == 0:
            cond = 'Fault ' + str(f.id)
        stop = ftriggered + 1
        # print('fault train: id=', f.id, 'DELAY=', f.DELAY)
        for s in range(numfeat):
            signal = XPlot[:, s]
            # print('start=', start, 'stop=', stop, 'cond=', cond)
            ax[s].plot(x[start:stop], signal[start:stop], color=color[i],
                       linestyle='-', linewidth=linewidth, label=cond)
            ax[s].axhline(y=NORMVAL[s], linestyle='--', linewidth=0.5,
                          label=None, color='green')

        start = stop - 1
        i += 1
        # print('BEFORE: cond=', cond, 'f.id=', f.id)
        if cond != 'normal':
            cond += '+' + str(f.id)
        else:
            cond = 'Fault ' + str(f.id)
    # print('BEFORE LAST: cond=', cond)

    stop = ntrain
    for s in range(numfeat):
        signal = XPlot[:, s]
        ax[s].plot(x[start:stop], signal[start:stop], color=color[i],
                   linestyle='-', linewidth=linewidth, label=cond)
        # print('start=', start, 'stop=', stop, 'label=', label)

    cond = 'normal'
    xoff = ntrain - 1
    t = xoff
    start = t
    i = 0
    faults = exp_test['faults']

    for f in faults:
        ftriggered = f.DELAY
        if ftriggered == 0:
            cond = 'Fault ' + str(f.id)
        stop = xoff + ftriggered + 1
        # print('fault train: id=', f.id, 'DELAY=', f.DELAY)
        for s in range(numfeat):
            signal = XPlot[:, s]
            # print('start=', start, 'stop=', stop)
            ax[s].plot(x[start:stop], signal[start:stop], color=color[i],
                       linestyle='-', linewidth=linewidth, label=cond)
        start = stop - 1
        i += 1
        if cond != 'normal':
            cond += '+' + str(f.id)
        else:
            cond = 'Fault ' + str(f.id)

    # start += 1
    stop = xoff + ntest
    # print('LAST: start=', start, 'stop=', stop, 'len(x)=', len(x))
    for s in range(numfeat):
        signal = XPlot[:, s]
        ax[s].plot(x[start:stop], signal[start:stop], color[i],
                   linestyle='-', linewidth=linewidth, label=cond)
        # print('s=', s, 'x=', x[start:stop], 'signal=', signal[start:stop])

    fig.suptitle(title, fontsize=fontsize)
    # The legend is relative to the whole figure, not relative
    # to the current axis
    handles, labels = ax[0].get_legend_handles_labels()
    # print('legend: handles=', handles, '\nlabels=', labels)

    fig.legend(handles, labels, fontsize=7,
               loc='upper center',  # 'lower left',
               # bbox_to_anchor=(0.0, 0.0),
               fancybox=True, shadow=True,
               ncol=nlegcol)
    plt.axis('tight')
    # input('...')

    if dropdir is not None:
        dt = datetime.now().strftime(
            '%Y_%m_%d__%H.%M.%S.%f')  # avoid ":"
        aux = (dropdir + dt + '_' + str(numfeat)
               + '_variable_train_test_signal_evolution.')
        dropfigfile = aux + figext
        plt.savefig(dropfigfile)
        print('CSTR.train_test_pair_signal_plot> Saving figure in ',
              dropfigfile)
        dropfigfile = aux + 'pdf'
        plt.savefig(dropfigfile)
        print('CSTR.train_test_pair_signal_plot> Saving figure in ',
              dropfigfile)
    # if save_fig_for_later_name is not None:
        # print('Saving plot for later use as ', save_fig_for_later_name)
        # save_obj(ax, save_fig_for_later_name) # must be stricly BEFORE plt.show
    plt.show(block=block)
