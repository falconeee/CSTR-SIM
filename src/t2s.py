import numpy as np
from scipy.linalg import sqrtm
from threshold import threshold
import matplotlib.pyplot as plt
from util import fig2file

def t2s(y, M, stat_type='t2',
        title=None, ax=None,
        color_intervals=None,
        block=True, saveas=None,
        no_labels=False):
    #title=None# DEBUG
    n, m = y.shape
    if M is None or n == 0 or m == 0:
        return None
    limvar = M['limvar']
    y_centered = y - M['mu']
    #zf = 0
    #resf = 0
    P = M['P']
    S = M['S']
    a = M['a']
    D = np.dot(np.dot(P, np.linalg.inv(S[:a, :a])), P.T)
    C = np.eye(m) - np.dot(P, P.T)
    #print('D=\n', D, 'shape=', D.shape)

    if stat_type == 't2':
        F = D
        limiar = threshold(M, stat_type)
        stat_str = '$T^{2}$'
        #print('limiar=', limiar)

    elif stat_type == 'q':
        stat_str = 'SPE'
        if a < m:
            F = C
            limiar = threshold(M, stat_type)
            #print('limiar=', limiar)
        else:
            Y = None
            auxstr = '\n'+str(a)+' principal components of '+str(m)+' possible variables'
            print('SPE (Q) fault detection statistic not applicable'+auxstr)
            return

    elif stat_type == 'c':
        stat_str = 'Combined'
        if a < m:
            limiar_t2 = threshold(M, 't2');
            limiar_q = threshold(M, 'q');
            F = D/limiar_t2 + C/limiar_q;
            limiar = threshold(M, stat_type)
            #print('threshold=', limiar)
        else:
            Y = None
            auxstr = '\n'+str(a)+' principal components of '+str(m)+' possible variables'
            print('Combined fault detection statistic not applicable'+auxstr)
            return

    else:
        raise Exception('Fault detection statistic ' + stat_type + ' unknown. Exit ...')

    z = np.zeros(n)
    for i in range(n): 
        yn = y_centered[i,:]
        z[i] = np.dot(yn, np.dot(F, yn.T)) # Computes stats
    zmean = z.mean()
    '''
    print('stat_type=', stat_type, 'a=', a, 'm=', m,
          '\nD=\n', D, 'shape=', D.shape,
          '\nC=\n', C, 'shape=', C.shape,
          '\nF=\n', F, 'shape=', F.shape) ; input('...')
    '''
    Y = {}
    Y['n'] = n  # how many samples used? Maybe nedded later, when z key is stripped
    Y['m'] = m  # number of variables of input data 
    Y['z'] = z
    Y['thr'] = limiar
    Y['zmean'] = zmean
    Y['in_control'] = zmean <= limiar
    above_thresh_qtd = sum(z>limiar)
    Y['F'] = F  # F is symmetric
    Y['F_sqrt'] = sqrtm(F)  # Store also the square root matrix
    #print('Is F symmetric?', np.linalg.norm(F-F.T), end=''); input(' = 0?'); #raise Exception()
    Y['FP'] = above_thresh_qtd  # this is not 'False positives, but faults detected
    Y['above'] = above_thresh_qtd/n
    Y['stat_type'] = stat_type
    Y['stat_str'] = stat_str
    Y['legs'] = None   # legends of plots

    #print('z=\n', z, 'shape=', z.shape)
    if title is not None:
        linewidth = 0.5
        fontsize = 7
        #print('ax=', ax); input('...')
        standalone = ax == None
        #fig = plt.gcf()
        label = 'Fault detection statistic='+stat_str+\
                 ' ('+str(a)+' of '+str(m)+' principal components, percentile='\
                +'{:.2f}'.format(limvar)+')'
        label_thr = ('PC '+str(a)+' of '+str(m)+' - Threshold='+'{:.3f}'.format(limiar))
        if standalone:
            ax = plt.gca()
        else:
            strstatval = stat_str+' mean of '+str(n)+' samples='+'{:.3f}'.format(zmean)
            strcontrol = ' --- in control' if Y['in_control'] else ' --- out of control'
            str_mean_control = strstatval+strcontrol
            #str_mean_control = ''

            #label = label_thr+'\n'+str_mean_control if not no_labels else None
            label = str_mean_control if not no_labels else None

        legs = []
        l = ax.axhline(y=limiar, linestyle='-', linewidth=linewidth,
                       label=label_thr, color='red')
        legs.append(l)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)
        if not no_labels:
            ax.set_xlabel('Sample', fontsize=fontsize)
        if color_intervals is not None:
            numintvals = len(color_intervals)
            for i in range(numintvals):
                start, stop, col, clabel = color_intervals[i]
                if i < numintvals-1:
                    stop += 1
                numpts = stop - start
                x = np.linspace(start, stop-1, numpts)
                #print('\ni=', i, 'numintvals=', numintvals, 'numpts=', numpts, 'x=', x)
                l, = ax.plot(x, z[start:stop], color=col, label=clabel, linewidth=linewidth)
                legs.append(l)

                #print('start=', start, 'stop=', stop,
                #      'numpts=', numpts, 'len(x)=', len(x), 'len(z[start:stop])=', len(z[start:stop])); #input('...')
        else:
            l, = ax.plot(z, linewidth=linewidth, label=label)
            legs.append(l)
        if not no_labels:
            ax.set_title(title, fontsize=fontsize)
            ax.legend(fontsize=fontsize, loc='upper left')#loc='best')
        if saveas is not None:
            fn = saveas + '.pdf'
            print('Saving the t2s graph as %s ...' % fn)
            # Save just the portion _inside_ the axis's boundaries
            fig = plt.gcf()
            extent = ax.get_tightbbox().transformed(transform=fig.dpi_scale_trans.inverted())
            #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # Padding
            fig.savefig(fn, bbox_inches=extent)
            #pad_x, pad_y = 1.2, 1.5
            #pad_x, pad_y = 1.0, 1.0
            #fig.savefig(fn, bbox_inches=extent.expanded(pad_x, pad_y))
        if standalone:
            plt.show(block=block)
        #print('t2s> legs=', legs); input('...')
        Y['legs'] = legs

    return Y
