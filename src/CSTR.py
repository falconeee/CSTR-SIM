import sys
import numpy as np
import copy
import struct
import subprocess
import warnings
import fortranformat as ff
from src.CSTR_plot import plot_signals, plotscatter


class Fault():
    """Fault class."""

    def __init__(self,
                 id=None,  # identifier for fault
                           # non-sensor (1,..) and sensor faults (2,3,...)
                 is_sensor_fault=False,
                 sensor_fault_type='bias',  # alternative ='value'
                 EXTENT0=None,
                 DELAY=None,
                 TC=None,  # Time constant
                 verbose=False):

        self.id_min = 2
        self.id_max = 22
        intvalstr = 'Must be in ['+str(self.id_min)+','+str(self.id_max)+'].'
        self.id_sensor_min = 1
        self.id_sensor_max = 14
        intvalsensorstr = ('(Must be in [' + str(self.id_min) +
                           ','+str(self.id_max) + '].)')

        if id is not None:
            if EXTENT0 is None:
                raise Exception('Fault must have extent value.')
            if DELAY is None:
                raise Exception('Fault must have delay value.')
            if TC is None:
                raise Exception('Fault must have time constant value.')
            if is_sensor_fault:
                if not (self.id_sensor_min <= id <= self.id_sensor_max):
                    raise Exception('Invalid sensor fault id. ' + intvalsensorstr)
            else:
                if not (self.id_min <= id <= self.id_max):
                    raise Exception('Invalid fault id. ' + intvalstr)
        else:
            DELAY = np.inf

        self.id = id
        self.is_sensor_fault = is_sensor_fault
        self.sensor_fault_type = sensor_fault_type
        self.EXTENT0 = EXTENT0
        self.DELAY = DELAY
        self.TC = TC
        self.DEXT = 0.0
        self.DEXT0 = 0.0


acronyms = ('c_A0', 'Q_1', 'T_1', 'L',
            'c_A', 'c_B', 'T_2',
            'Q_5', 'Q_4', 'T_3', 'h_7',
            'u_1',
            'u_3=m_2',
            'u_2',
            'r_1', 'r_2', 'r_3', 'r_4',
            'CLASS')


def print_acronyms(file=sys.stdout, truncate=None):
    print(';'.join(acronyms), file=file)


class CSTR():
    """CSTR class."""

    def __init__(self, id=None,
                 controller_parm=np.array(((35.0, 5.0, 0.0),  # LEVEL
                                           (-0.040, -0.020, 0.0),  # TEMP
                                           (-25.0, -75.0, 0.0))),  # COOLFLOW
                 theta=1.0,  # EWMA (EXP. FILTER) PARMETER
                 randseed=1234,
                 fortran_rand=False,
                 faults=(),
                 timehoriz=100,
                 verbose=False):

        self.id = id
        self.controller_parm = controller_parm
        self.B = self.controller_parm
        self.THETA = theta
        self.DSEED = randseed
        self.maxerr = 3

        self.sensors = 14
        self.num_constraints = 4
        self.numvar = self.sensors + self.num_constraints
        self.faults, self.numfaults = list(faults), len(faults)
        self.classstr = ''
        self.verbose = verbose
        self.acronyms = acronyms

        self.datarootdir = './data/'
        if self.id is None:
            self.datafn = self.datarootdir + 'X_py.csv'
        else:
            self.datafn = self.datarootdir + id + '.csv'
            
        self.ALPHA1 = 2500.0
        self.BETA1 = 25000.0
        self.ALPHA2 = 3000.0
        self.BETA2 = 45000.0
        self.CA0 = 20.0
        self.CA00 = copy.copy(self.CA0)
        self.CA = 2.850
        self.CB = 17.114
        self.CCNOM = 0.0226
        self.CC = self.CCNOM
        self.CP = 4.2
        self.CP1 = copy.copy(self.CP)
        self.CP2 = copy.copy(self.CP)
        self.DT = 0.02
        self.FINTEG = 0.0
        self.JEP = 0.0
        self.MOLIN = 0.0
        self.MOLOUT = 0.0
        self.PB = 2000.0
        self.PP = 48000.0
        self.PP0 = copy.copy(self.PP)
        self.QREAC1 = 30000.0
        self.QREAC2 = -10000.0
        self.QEXT = 0.0
        self.REP = 0.0
        self.RHO = 1000.0
        self.RHO1 = copy.copy(self.RHO)
        self.RHO2 = copy.copy(self.RHO)
        self.S3 = False
        self.S4 = False
        self.LEVEL = 2.0
        self.TAREA = 1.5
        self.TIME = 0.0
        self.MINUTES = 0
        self.timehoriz = timehoriz
        self.iter = 0
        self.smppermin = int(1.0 / self.DT)
        self.maxiter = self.timehoriz * self.smppermin

        self.UA = 1901.0
        self.VOL = self.TAREA * self.LEVEL
        self.VOL0 = copy.copy(self.VOL)

        self.PCW = 56250.0
        self.CNT = np.array([74.7, 0.9, 59.3])
        self.FLOW = np.array([0.25, 0.25, 0.0, 0.25, 0.9, 0.0, 0.0, 0.9])
        self.SP = np.array([2.0, 80.0, 0.9])
        self.SDEV = np.array([0.15, 0.002, 0.15, 0.01, 0.02, 0.14,
                              0.15, 0.003, 0.002, 0.15, 400.0, 0.01,
                              0.01, 0.0001, 0.005, 0.005, 0.01, 0.5])
        self.ERROR = np.zeros(3)
        self.DERROR = np.zeros(3)
        self.R = np.array([100.0, 1e6, 0.0, 500.0, 72.0, 0.0, 1e6,
                           1e6, 0.0, 65.0])
        self.R0 = copy.copy(self.R)
        self.RCOMP = np.zeros(10)
        self.REG1 = np.zeros(3)
        self.REG2 = np.zeros(3)
        self.T = np.array([30.0, 80.0, 20.0, 40.0])
        self.V = np.array([74.7, 59.3])

        self.MEAS1 = np.array([20.0, 0.25, 30.0, 2.00, 2.85, 17.11,
                               80.0, 0.9, 0.25, 20.0, 56250.0, 25.3, 40.7,
                               0.9, 0.0, 0.0, 0.0, 0.0])
        self.MEAS2 = copy.copy(self.MEAS1)
        self.NORMVAL = copy.copy(self.MEAS1)

        self.MASSBAL = 0.0
        self.EPD = 0.0
        self.CWPD = 0.0
        self.MOLBAL = 0.0

        self.RE, self.RC = np.nan, np.nan
        self.randbuf = None
        self.fortran_rand = fortran_rand
        if self.fortran_rand:
            from rand_fortran import GGNML
            self.GGNML = GGNML
        else:
            np.random.seed(seed=self.DSEED)
            
        self.dbgvar = np.zeros(10)
        self.dbg_meas_sum = np.zeros(100)
        self.dbg_meas_sum = copy.copy(self.NORMVAL)
        self.dbg_flow_sum = copy.copy(self.FLOW)

    def open(self):
        self.outfile = open(self.datafn, mode='w')
        print_acronyms(self.outfile)

    def close(self):
        self.outfile.close()

    def fortran_random_generation(self):
        DSEED = self.DSEED
        MAXSTEPS = self.maxiter
        fncfg = 'rand_deterministic.cfg'
        with open(fncfg, mode='w') as f:
            f.write(str(DSEED)+'\n')
            f.write(str(MAXSTEPS)+'\n')
            f.close()

        prog_fortran = 'rand_deterministic'
        subprocess.run([prog_fortran])

        with open('rand_deterministic.bin', mode='rb') as f:
            data = f.read()
            f.close()

        nbytes = len(data)
        self.randbuf = np.zeros(MAXSTEPS*(self.numvar+3))
        cnt = 0
        for i in range(int(nbytes/8)):
            d = data[8*i:8*i+8]
            value = struct.unpack('<d', d)[0]
            self.randbuf[cnt] = value
            cnt += 1

    def update_affected_param(self, S0):
        f = self.faults[S0]
        if self.TIME < f.DELAY:
            return

        fnr = f.id
        if f.is_sensor_fault:
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            return

        if fnr == 2:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.R[0]
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.R[0] = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 3:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.R[8]
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.R[8] = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 4:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.R[7]
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.R[7] = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 5:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.R[6]
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.R[6] = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 6:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.R[1]
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.R[1] = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 7:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.PP
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.PP = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 8:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.UA
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.UA = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 9:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.QEXT
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.QEXT = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 10:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.BETA1
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.BETA1 = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 11:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.BETA2
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.BETA2 = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 12:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.FLOW[0]
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.FLOW[0] = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 13:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.T[0]
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.T[0] = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 14:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.CA0
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.CA0 = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 15:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.T[2]
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.T[2] = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 16:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.PCW
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.PCW = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 17:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.JEP
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.JEP = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 18:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.REP
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.REP = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 19:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.SP[0]
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.SP[0] = f.EXTENT0 - f.DEXT0 * f.DEXT

        if fnr == 20:
            if f.DEXT0 == 0.0:
                f.DEXT0 = f.EXTENT0 - self.SP[1]
            f.DEXT = np.exp(f.TC * (f.DELAY - self.TIME))
            self.SP[1] = f.EXTENT0 - f.DEXT0 * f.DEXT

        self.faults[S0] = f

    def update_controllers(self):
        MEAS = [self.MEAS1[3], self.MEAS1[6], self.MEAS1[7]]
        B = self.controller_parm

        for i in range(3):
            self.REG2[i] = self.REG1[i]
            self.REG1[i] = self.DERROR[i]
            dif = self.SP[i] - MEAS[i]
            self.DERROR[i] = self.ERROR[i] - dif
            self.ERROR[i] -= self.DERROR[i]

            aux = (self.CNT[i] - B[i][0] * self.DERROR[i]
                   + B[i][1] * (0.5 * self.DERROR[i] + self.ERROR[i]) * self.DT
                   + B[i][2] * (2.0 * self.REG1[i] - 0.5*self.REG2[i] -
                                1.5 * self.DERROR[i]) / self.DT)
            aux1 = max(0.0, aux)
            if i == 1:
                self.CNT[i] = aux1
            else:
                self.CNT[i] = min(100.0, aux1)

    def evaluate_safety_systems(self):
        MEAS41 = self.MEAS1[3]
        MEAS71 = self.MEAS1[6]
        minlevel = 1.2
        maxlevel = 2.75
        maxtemp = 130.0

        if MEAS41 >= maxlevel or MEAS71 >= maxtemp:
            if not self.S3:
                for i in range(self.numfaults):
                    f = self.faults[i]
                    if not f.is_sensor_fault and f.id == 12:
                        self.faults[i].DEXT = 0.0
                        self.faults[i].DEXT0 = 0.0
                self.FLOW[0] = 0.0
                self.S3 = True
            reason = ''
            if MEAS41 >= maxlevel:
                reason += 'REACTOR LEVEL AT ' + '{:5.2f}'.format(MEAS41)
            if MEAS71 >= maxtemp:
                reason += 'REACTOR TEMP  ' + '{:5.2f}'.format(MEAS71)
            warnstr = ('***** EMERGENCY SHUTDOWN INITIATED AT ' +
                       '{:5.2f}'.format(self.TIME) + ' --- REASON: ' + reason)
            warnings.warn(warnstr)
            self.shutdown = True
            self.shutdown_iter = self.iter
            self.shutdown_TIME = self.TIME
            self.shutdown_MINUTES = self.MINUTES

        elif MEAS41 <= minlevel:
            if not self.S4:
                for i in range(self.numfaults):
                    f = self.faults[i]
                    if not f.is_sensor_fault and f.id == 7:
                        self.faults[i].DEXT = 0.0
                        self.faults[i].DEXT0 = 0.0
                self.PP = 0.0
                self.S4 = True
            reason = 'REACTOR LEVEL AT ' + '{:5.2f}'.format(MEAS41)
            warnstr = ('***** LOW LEVEL FORCES PUMP SHUTDOWN AT ' +
                       '{:5.2f}'.format(self.TIME) + ' --- REASON: ' + reason)
            warnings.warn(warnstr)
            self.shutdown = True
            self.shutdown_TIME = self.TIME
            self.shutdown_MINUTES = self.MINUTES
        else:
            pass

    def calc_valve_positions(self):
        if self.numfaults == 0:
            self.V[0] = min(100.0, max(0.0, 100.0 - self.MEAS2[11]))
            self.V[1] = min(100.0, max(0.0, 100.0 - self.MEAS2[12]))
        else:
            for i in range(self.numfaults):
                f = self.faults[i]
                if (not f.is_sensor_fault and f.id == 21
                        and self.TIME >= f.DELAY):
                    self.V[0] = 100.0 - f.EXTENT0 * (1.0 - f.DEXT)
                else:
                    self.V[0] = min(100.0, max(0.0, 100.0 - self.MEAS2[11]))

                if (not f.is_sensor_fault and f.id == 22
                        and self.TIME >= f.DELAY):
                    self.V[1] = 100.0 - f.EXTENT0 * (1.0 - f.DEXT)
                else:
                    self.V[1] = min(100.0, max(0.0, 100.0 - self.MEAS2[12]))

    def calc_flow_rates(self):
        self.R[2] = 5.0 * np.exp(0.0545*self.V[0])
        self.R[5] = 5.0 * np.exp(0.0545*self.V[1])

        self.RE = (1/((1/self.R[1])+(1/(self.R[2]+self.R[3]))))+self.R[0]
        self.RC = ((1/((1/self.R[6])+(1/self.R[7]) +
                       (1/(self.R[8]+self.R[9]))))+self.R[4]+self.R[5])

        aux = (self.PP + self.PB - self.REP)
        if aux <= 0:
            self.FLOW[1] = 0.0
        else:
            self.FLOW[1] = np.sqrt(aux) / self.RE

        aux -= (self.FLOW[1] * self.R[0])**2.0
        if aux <= 0:
            self.FLOW[2] = 0.0
        else:
            self.FLOW[2] = np.sqrt(aux) / self.R[1]
        self.FLOW[3] = self.FLOW[1] - self.FLOW[2]

        self.FLOW[4] = np.sqrt(self.PCW - self.JEP) / self.RC

        aux = self.PCW - self.JEP - (self.FLOW[4]*(self.R[4]+self.R[5]))**2.0
        if aux <= 0:
            self.FLOW[5] = 0.0
            self.FLOW[6] = 0.0
        else:
            auxsqrt = np.sqrt(aux)
            self.FLOW[5] = auxsqrt / self.R[6]
            self.FLOW[6] = auxsqrt / self.R[7]
        self.FLOW[7] = self.FLOW[4] - self.FLOW[5] - self.FLOW[6]

    def calc_thermo_level_volume(self):
        self.T[3] = ((self.UA * self.T[1] + self.RHO2 *
                     self.CP2 * self.FLOW[7] * self.T[2]) /
                     (self.UA + self.RHO2 * self.CP2 * self.FLOW[7]))
        self.QJAC = self.UA * (self.T[1] - self.T[3])

        self.VOLD = self.VOL
        self.RHOLD = self.RHO
        self.CPOLD = self.CP

        self.VOL = (self.VOLD + (self.FLOW[0] + self.FLOW[5] - self.FLOW[1]) *
                    self.DT)
        iVOL = 1 / self.VOL
        self.LEVEL = self.VOL / self.TAREA
        if self.VOL <= 0.0:
            warnstr = ('***** FAILURE DUE TO LOW LEVEL FORCES SHUTDOWN AT ' +
                       '{:5.2f}'.format(self.TIME))
            warnings.warn(warnstr)
            self.PB = 0.0
            self.shutdown = True
        else:
            self.RHO = (iVOL * (self.VOLD * self.RHO) +
                        iVOL *
                        (self.DT * (self.FLOW[0] *
                                    self.RHO1 + self.FLOW[5] *
                                    self.RHO2 - self.FLOW[1] *
                                    self.RHO)))

            self.CP = (iVOL * (self.VOLD * self.CP) +
                       iVOL *
                       (self.DT * (self.FLOW[0] * self.CP1 +
                                   self.FLOW[5] * self.CP2 -
                                   self.FLOW[1] * self.CP)))

            self.PB = self.RHO * self.LEVEL

        self.RRATE1 = (self.ALPHA1 * np.exp(
            -self.BETA1 / (8.314 * (273.15 + self.T[1]))) * self.CA)
        self.RRATE2 = (self.ALPHA2 * np.exp(
            -self.BETA2 / (8.314 * (273.15 + self.T[1]))) * self.CA)

        self.CA = (iVOL * (self.VOLD * self.CA) +
                   iVOL * (self.FLOW[0] * self.CA0 -
                           self.FLOW[1] * self.CA - self.RRATE1 *
                           self.VOLD - self.RRATE2 * self.VOLD) *
                   self.DT)

        self.CB = (iVOL * (self.VOLD * self.CB) +
                   iVOL * (self.RRATE1 * self.VOLD -
                           self.FLOW[1] * self.CB) * self.DT)
        self.CC = (iVOL * (self.VOLD * self.CC) +
                   iVOL * (self.RRATE2 * self.VOLD -
                           self.FLOW[1] * self.CC) * self.DT)

        aux = self.VOL * self.RHO * self.CP
        iaux = 1 / aux
        self.T[1] = (iaux * (self.VOLD * self.RHOLD * self.CPOLD * self.T[1]) +
                     iaux * (((self.QREAC1 * self.RRATE1 +
                               self.QREAC2 * self.RRATE2) *
                             self.VOLD) * self.DT) +
                     iaux * ((self.QEXT - self.QJAC) * self.DT) +
                     iaux * (self.FLOW[0] * self.RHO1 * self.CP1 *
                             self.T[0] * self.DT) +
                     iaux * (self.FLOW[5] * self.RHO2 * self.CP2 *
                             self.T[3] * self.DT) -
                     iaux * (self.FLOW[1] * self.RHOLD * self.CPOLD *
                             self.T[1] * self.DT))

    def measure(self):
        self.MEAS2[0] = self.CA0
        self.MEAS2[1] = self.FLOW[0]
        self.MEAS2[2] = self.T[0]
        self.MEAS2[3] = self.LEVEL
        self.MEAS2[4] = self.CA
        self.MEAS2[5] = self.CB
        self.MEAS2[6] = self.T[1]
        self.MEAS2[7] = self.FLOW[4]
        self.MEAS2[8] = self.FLOW[3]
        self.MEAS2[9] = self.T[2]
        self.MEAS2[10] = self.PCW
        self.MEAS2[11] = 100.0 - self.CNT[0]
        self.MEAS2[12] = 100.0 - self.CNT[2]
        self.MEAS2[13] = self.CNT[1]

        for s in range(self.sensors):
            if self.fortran_rand:
                self.DSEED, RAND = self.GGNML(self.DSEED, 3)
            else:
                RAND = np.random.normal(size=3)

            if self.numfaults == 0:
                self.MEAS2[s] += RAND[0] * self.SDEV[s]
            else:
                for i in range(self.numfaults):
                    f = self.faults[i]
                    aux = RAND[i] * self.SDEV[s]
                    if f.is_sensor_fault and self.TIME >= f.DELAY:
                        if s+1 == f.id and f.sensor_fault_type == 'bias':
                            self.MEAS2[s] += (aux + f.EXTENT0 * (1 - f.DEXT))
                        elif s+1 == f.id and f.sensor_fault_type == 'value':
                            self.MEAS2[s] = f.EXTENT0 * (1 - f.DEXT)
                        else:
                            self.MEAS2[s] += aux
                    else:
                        self.MEAS2[s] += aux

        for s in range(self.sensors):
            self.MEAS1[s] = (self.THETA*self.MEAS2[s] +
                             (1.0 - self.THETA)*self.MEAS1[s])

    def eval_constraints(self):
        difFLOW1FLOW4 = self.MEAS2[1] - self.MEAS2[8]
        self.FINTEG += (difFLOW1FLOW4 + self.FLOW[5] - self.FLOW[2]) * self.DT

        VOLMEAS = self.TAREA * self.MEAS2[3]
        VOLDIF = VOLMEAS - self.VOL0
        self.MASSBAL = VOLDIF - self.FINTEG

        self.PBCOMP = self.RHO1 * self.MEAS2[3]

        self.RCOMP[2] = 5.0 * np.exp(0.0545 * (100.0 - self.MEAS2[11]))

        aux = np.sqrt(self.PBCOMP + self.PP0)
        self.EPD = (self.MEAS2[8] - ((1.0 / (self.RCOMP[2] +
                                             self.R0[0] + self.R0[3]))*aux))

        self.RCOMP[5] = 5.0 * np.exp(0.0545 * (100.0 - self.MEAS2[12]))
        aux = self.RCOMP[5] + self.R0[4] + self.R0[8] + self.R0[9]
        self.CWPD = self.MEAS2[7] - (1.0 / aux) * np.sqrt(self.MEAS2[10])

        sumConcABC = self.MEAS2[4] + self.MEAS2[5] + self.CCNOM
        dmolin = self.MEAS2[0] * self.MEAS2[1] * self.DT
        dmolout = sumConcABC * (self.MEAS2[8] + self.FLOW[2]) * self.DT
        self.MOLIN += dmolin
        self.MOLOUT += dmolout
        mol = sumConcABC * VOLMEAS
        self.MOLBAL = mol - self.CA00 * self.VOL0 - self.MOLIN + self.MOLOUT

        self.MEAS2[14] = self.MASSBAL
        self.MEAS2[15] = self.CWPD
        self.MEAS2[16] = self.EPD
        self.MEAS2[17] = self.MOLBAL

    def set_classstr(self):
        if self.shutdown:
            self.classstr = 'shutdown'
            return

        active_faults = 0
        for i in range(self.numfaults):
            if self.TIME >= self.faults[i].DELAY:
                active_faults += 1

        self.classstr = 'normal'
        if active_faults > 0:
            self.classstr = ''
            for i in range(active_faults):
                f = self.faults[i]
                if f.is_sensor_fault:
                    self.classstr += 'S'
                else:
                    self.classstr += ''
                self.classstr += str(f.id)
                if i < active_faults-1:
                    self.classstr += '+'

    def measure_out(self):
        M = self.MEAS2
        lineformat = ff.FortranRecordWriter('(E15.5)')
        
        for s in range(self.numvar):
            mstr = lineformat.write([M[s]])
            # Imprime a string limpa (.strip()) seguida do ponto e vírgula
            print(mstr.strip(), file=self.outfile, end=';')
            
        self.set_classstr()
        print(self.classstr, file=self.outfile)

    def peepvars(self):
        pass

    def run(self):
        print('Running Data Genaration...')
        done = False
        self.shutdown = False

        self.measure_out()

        while not done and not self.shutdown:
            for S0 in range(self.numfaults):
                self.update_affected_param(S0)

            self.update_controllers()
            self.evaluate_safety_systems()
            self.calc_valve_positions()
            self.calc_flow_rates()
            self.calc_thermo_level_volume()
            self.measure()
            self.eval_constraints()

            self.iter += 1
            self.TIME += self.DT
            self.SP[2] = self.MEAS2[13]

            if self.iter % self.smppermin == 0 or self.shutdown:
                self.measure_out()
                self.MINUTES += 1

            done = self.iter >= self.maxiter

            if False:
                for s in range(self.numvar):
                    self.dbg_meas_sum[s] += self.MEAS2[s]
                for i in range(len(self.FLOW)):
                    self.dbg_flow_sum[i] += self.FLOW[i]
                self.dbgvar[0] += self.MEAS2[1] - self.MEAS2[8]
                self.dbgvar[1] += self.FINTEG
                self.dbgvar[2] += self.TAREA * self.MEAS2[3] - self.VOL0
                self.dbgvar[3] += self.VOL
                self.dbgvar[4] += self.MEAS2[14]
                self.dbgvar[5] += self.FLOW[0] - self.FLOW[3]

        print('Data Genaration successfully completed!')


def run_experiment(experiment, do_run=True, do_plot=True):
    e = experiment
    cstr = CSTR(id=e['id'],
                theta=e['theta'], randseed=e['randseed'],
                fortran_rand=e['fortran_rand'],
                faults=e['faults'],
                timehoriz=e['timehoriz'])

    if do_run:
        cstr.open()
        cstr.run()
        cstr.close()
    
    feat1 = 1
    feat2 = 11
    feat3 = 14
    mask = e['plotmask']
    block = True

    if do_plot:
        plot_signals(cstr, mask=mask, block=block)

    feat3 = None

    if do_plot:
        azim = -22; elev = 12
        azim = -24; elev = 7
        plotscatter(cstr, feat1=feat1, feat2=feat2, feat3=feat3,
                    standardize=True,
                    dropfigfile=cstr.datarootdir+str(e['id']),
                    title='CSTR: Condition in Feature Space',
                    block=block, azim=azim, elev=elev)
    return cstr