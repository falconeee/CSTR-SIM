import numpy as np
from scipy.stats import chi2
from scipy.stats import f as F_distribution
from scipy.stats import norm


def threshold(M, stat_type):
    a = M['a']
    n = M['n']
    alfa = M['alfa']
    mu = M['mu']
    m = len(mu)

    if stat_type == 't2' or stat_type == 'c':
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html#scipy.stats.f
        # Percent point function (inverse of cdf — percentiles)
        t2 = (a*(n-1)*(n+1)/(n*(n-a))) * F_distribution.ppf(q=alfa, dfn=a, dfd=n-a) # Braatz

    if stat_type == 'q' or stat_type == 'c':
        S = M['S']
        D = np.diag(S)
        d = D[a:m]
        if len(d) == 0:
            stat = None
            return stat

        teta1 = sum(d)
        teta2 = sum(d**2)
        teta3 = sum(d**3)
        h0 = 1-(2*teta1*teta3/(3*teta2**2))
        # https://stackoverflow.com/questions/20626994/how-to-calculate-the-inverse-of-the-normal-cumulative-distribution-function-in-p
        ca = norm.ppf(alfa)
        Q = (ca * np.sqrt(2*teta2*h0**2) / teta1 + 1 + teta2*h0*(h0-1) / teta1**2)
        Qstat = teta1*Q**(1.0/h0)
        # print('d=\n', d, '\nteta1=', teta1, 'teta2=', teta2, 'teta3=',
        #      teta3, 'h0=', h0, 'alfa=', alfa, 'ca=', ca, 'Q=', Q, 'Qstat=', Qstat) 

    if stat_type == 't2':
        stat = t2

    elif stat_type == 'q':
        stat = Qstat

    elif stat_type == 'c':
        # g1 = (t2**(-2) + teta2*Q**(-2)) / (t2**(-1)+teta1*Q**(-1))
        g1 = (t2**(-2) + teta2 * Qstat**(-2)) / (t2**(-1) + teta1*Qstat**(-1))
        h1 = (t2**(-1) + teta1 * Qstat**(-1))**2 / (t2**(-2) + teta2*Qstat**(-2))
        stat = g1 * chi2.ppf(alfa, h1)
        # print('stat_type=', stat_type, 'Q=', Q, 'Qstat=', Qstat, 't2=',
        #       t2, 'g1=', g1, 'h1=', h1, 'stat=', stat)

    return stat
