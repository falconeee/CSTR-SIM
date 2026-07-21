from __future__ import print_function

import numpy as np

# http://arogozhnikov.github.io/2015/09/30/NumpyTipsAndTricks2.html
def running_average_cumsum(seq, window=100):

    '''
    n1 = 1. / window
    
    cumsum = np.cumsum(seq)
    s = np.insert(cumsum, 0, [0])
    print('seq=', seq, '\ncumsum=', cumsum, '\ns=', s)
    right = s[window:]
    left = s[:-window]
    diff = right - left
    print('right=', right, '\n left=', left, '\n diff=', diff)
    means = diff * n1
    print('means=', means)


    seqlen = len(seq)
    assert seqlen > window, 'Window must be smaller than sequence'


    print('\n\n\nseq=', seq)
    s = seq[0:window]
    cumsum = np.cumsum(s)
    s = np.insert(cumsum, 0, [0])
    print('Init: s=', s, '\ncumsum=', cumsum)
    
    done = False
    i = window
    lidx = window
    ridx = -window-1
    while not done:
        
        right = s[lidx]
        left = s[ridx]
        diff = right - left
        mean = diff * n1
        print('i=', i, 'left=', left, 'right=', right,
              'diff=', diff, 'mean=', mean, 's=', s)

        done = i == seqlen
        if not done:
            cs = right+seq[i]
            s = np.insert(s, window+1, [cs])
            s = np.delete(s, 0)
            i += 1
    return means
    '''



    seqlen = len(seq)
    assert seqlen > window, 'Window must be smaller than sequence'
    n1 = 1. / window

    sinit = seq[0:window-1]
    cumsum = np.cumsum(sinit)
    sinit = np.insert(cumsum, 0, [0])
    print('Init: s=', sinit, '\ncumsum=', cumsum)

    s = np.zeros(seqlen+1)
    s[0:window] = sinit
    l = 0
    r = window
    print('s=', s, '\ns=', s)
            
    done = False
    while not done:
        done = r == seqlen+1
        if not done:
            s[r] = seq[r-1] + s[r-1]
            d = s[r]- s[l]   
            m = d*n1
            print('>>>>> l=', l, 'r=', r, 'd=', d, 'm=', m, 's=', s)
            r += 1
            l += 1



def detect_fault(seq, thresh, window=100):
    '''Given a signal sequence and a window size, test if the mean
    of the signal in the window surpasses a threshold.
    If a fault is detected return the value of the window mean and the position
    '''
    seqlen = len(seq)
    assert seqlen > window, 'Window must be smaller than sequence'
    n1 = 1. / window

    sinit = seq[0:window-1]
    cumsum = np.cumsum(sinit)
    sinit = np.insert(cumsum, 0, [0])
    print('Init: sinit=', sinit, '\nseq=', seq)

    s = np.zeros(seqlen+1)
    s[0:window] = sinit
    l = 0
    r = window
    print('s=', s, '\ns=', s)
            
    done = False
    fault = False
    while not done:
        done = r == seqlen+1 or fault
        if not done:
            s[r] = seq[r-1] + s[r-1]
            d = s[r] - s[l]
            m = d*n1
            fault = m >= thresh
            print('>>>>> fault=', fault, 'l=', l, 'r=', r, 'd=', d, 'm=', m, 's=', s)
            if not fault:
                r += 1
                l += 1

    return fault, m, l


seq = np.array([2, 1, 6, 2, 1, 4, 1, 2, 1])
window = 3

seq = np.array([2, 1, 2, 4, 6, 4, 7, 3, 2, 1, 2])
window = 3

is_fault, index, pos = detect_fault(seq, thresh=4.3, window=window)

print('Fault=', is_fault, 'index=', index, 'pos=', pos)


#ra = running_average_cumsum(seq, window)
#print('\nra=', ra)
