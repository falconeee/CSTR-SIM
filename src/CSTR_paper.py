# +++ P A P E R +++ #

import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.LabelBinarizer2 import LabelBinarizer2
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt

import src.CSTR
from src.CSTR import Fault, run_experiment
from src.CSTR_plot_statistics import plot_2_D_statistics, plot_stat, join_figs
from src.CSTR_plot import read_X, dataframe2sklearn, plot_signals, plot_tSNE, plot_PCA


dropdir = './output/'


def visualize():
    #sys.path.append('/home/thomas/Nextcloud/book/soft/lib/')
    from CNN import CNN
    scal = 5
    ex1 = {'id': 'paper1', 'theta': 1, 'randseed': 1234, 'fortran_rand': False,
           'timehoriz': 100*scal,
           'plotmask': None,  # [1, 3, 4, 7, 8, 11, 12, 13, 14, 16],
           'faults': (
                      Fault(id=3, EXTENT0=50, DELAY=50*scal, TC=0.1),
                      Fault(is_sensor_fault=True, id=2, sensor_fault_type='bias',
                            EXTENT0=0.01, DELAY=70*scal, TC=0.01)
                     )
          }
    cstr = run_experiment(ex1, do_run=True, do_plot=True)
    datafn = cstr.datafn
    print('Opening input data file %s.' % datafn)
    df = read_X(datafn=datafn)
    X, y, featname = dataframe2sklearn(df)
    plot_2_D_statistics(X, stat_type='t2', alfa=0.95, limvar=0.95,
                        force_a=2, plot_subspace=True)


def fault_detect():

    scal = 10
    ex2 = {'id': 'paper2', 'theta': 1, 'randseed': 1234, 'fortran_rand': False,
           'timehoriz': 100*scal,
           'plotmask': None,  # [1, 3, 4, 7, 8, 11, 12, 13, 14, 16],
           'faults': (
                      # Fault(id=2, EXTENT0=150, DELAY=50*scal, TC=0.1),
                      Fault(id=2, EXTENT0=150, DELAY=50*scal, TC=0.1),
                     )
          }
    cstr = run_experiment(ex2, do_run=True, do_plot=True)
    datafn = cstr.datafn
    print('Opening input data file %s.' % datafn)
    df = read_X(datafn=datafn)
    X, y, featname = dataframe2sklearn(df)
    print(X)
    print(y)
    # raise Exception

    force_a = 3
    force_a = None
    alfa = 0.95
    limvar = 0.95
    for stat_type in ('t2', 'q', 'c'):
        plot_stat(cstr, stat_type=stat_type, alfa=alfa, limvar=limvar,
                  force_a=force_a,
                  dropfigfile=dropdir + 'stat_' + stat_type)


def fault_diagnose(standardize=True, do_run=True, do_tSNE=True):
    from CSTR_machine_learning import CNN, CNN1

    scal = 100
    timehoriz = 10 * scal
    faults = (
        Fault(id=None),
        # Fault(id=2, EXTENT0=200, DELAY=0*scal, TC=0.1),  # Normal = 100
        # Fault(id=3, EXTENT0=100, DELAY=0*scal, TC=0.1),    # Normal = 0
        Fault(id=14, EXTENT0=21, DELAY=0*scal, TC=0.1),    # Normal = 20
        Fault(id=5, is_sensor_fault=True, EXTENT0=2.80, DELAY=0*scal, TC=0.1),  # Normal = 2.85
        # Fault(id=10, EXTENT0=24000, DELAY=0*scal, TC=0.1),  # Normal = 25000
        # Fault(id=11, EXTENT0=44000, DELAY=0*scal, TC=0.1),  # Normal = 45000
             )

    numfaults = len(faults)
    if do_run:
        for i in range(numfaults):
            f = faults[i]
            # print('i=', i, 'f=', f)
            cstr = CSTR.CSTR(id=str(f.id), faults=(f, ), timehoriz=timehoriz)
            cstr.open()
            cstr.run()
            cstr.close()
            # plot_signals(cstr) ; input('...')

    for i in range(numfaults):
        f = faults[i]
        print('i=', i, 'f=', f)
        cstr = CSTR.CSTR(id=str(f.id), timehoriz=timehoriz)
        datafn = cstr.datafn
        print('Opening input data file %s.' % datafn)
        df = read_X(datafn=datafn)
        X, y, featname = dataframe2sklearn(df)

        if i == 0:
            X_all, y_all = X, y
        else:
            X_all = np.append(X_all, X, axis=0)
            y_all = np.concatenate((y_all, y), axis=0)

    y_all = y_all.astype(str)
    print('X_all=\n', X_all, 'shape=', X_all.shape,
          '\ny_all=\n', y_all, 'shape=', y_all.shape)  # ; input('...')

    X, y = X_all, y_all

    if standardize:
        scaler = StandardScaler()

    if standardize:
        scaler.fit(X)
        if do_tSNE:
            X_tSNE = scaler.transform(X, copy=True)
    else:
        if do_tSNE:
            X_tSNE = X

    if do_tSNE:
        plot_tSNE(X_tSNE, y, n_components=3,
                  dropfigfile=cstr.datarootdir+'tSNE')
        return

    classname = np.unique(y)
    # ynum = LabelEncoder().fit_transform(y)
    dim_output = len(classname)

    n_samples, dim_input = X.shape


    lb = LabelBinarizer2()
    Y = lb.fit_transform(y)

    #  CNN
    y = np.resize(y, (n_samples,))
    chan = 1
    # X = np.resize(X, (X.shape[0], X.shape[1], 1))
    dim_output = len(np.unique(y))
    Y = lb.fit_transform(y)
    # print('y=\n', y, '\nY=\n', Y); input('...')

    kernel_size = 16    # size of the convolution filter (image would be e.g. tupel (3,3) )
    filters = 10  # number of convolution filters

    epochs = 10
    # epochs = 1  # DEBUG
    batch_size = 1
    verbose = 1

    model = CNN(dim_input=dim_input, chan=chan, dim_output=dim_output,
                filters=filters, kernel_size=kernel_size,
                batch_size=batch_size, epochs=epochs, verbose=verbose)
    model = CNN1(dim_input=dim_input, dim_output=dim_output)

    print('model.summary():\n')
    model.summary()
    # input('...')
    params = model.get_params()
    print('\n\nmodel=', model)
    print('params=', params)

    '''
    '''

    # model = KNeighborsClassifier(n_neighbors=3)

    K = 3
    # K = 2  # DEBUG
    skf = StratifiedKFold(n_splits=K, shuffle=True)

    y_pred_overall = []
    y_test_overall = []
    k = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        if standardize:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train, copy=True)
            X_test = scaler.transform(X_test, copy=True)

        X_train = np.resize(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.resize(X_test, (X_test.shape[0], X_test.shape[1], 1))

        y_train, y_test = y[train_index], y[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        print('X_train.shape=', X_train.shape, 'X_test.shape=', X_test.shape)
        print('K-Fold: Fold #', k+1, 'of', K)

        # model.seq.build(input_shape=(model.dim, model.chan))
        # model.seq.fit(X_train, Y_train, batch_size=batch_size,
        #               epochs=epochs, verbose=verbose)http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/

        model.fit(X_train, Y_train)

        y_pred = model.predict(X_test)
        y_pred_class = lb.inverse_transform(y_pred)
        # y_pred_class = model.seq.predict_classes(X_test)
        y_pred_overall = np.concatenate([y_pred_overall, y_pred_class])
        y_test_overall = np.concatenate([y_test_overall, y_test])
        k += 1

    print('CNN Classification Report: ')
    # print('y_test_overall=', y_test_overall,
    # 'y_pred_overall=', y_pred_overall, 'classname=', classname)
    print(classification_report(y_test_overall, y_pred_overall,
                                target_names=classname, digits=3))
    print('Accuracy=', '%.2f %%' %
          (100*accuracy_score(y_test_overall, y_pred_overall)))
    print('Macro-averaged F1=', '%.3f' %
          (f1_score(y_test_overall, y_pred_overall, average='macro')))
    print('Micro-averaged F1=', '%.3f' %
          (f1_score(y_test_overall, y_pred_overall, average='micro')))
    print('CNN Confusion Matrix: ')
    cm = confusion_matrix(y_test_overall, y_pred_overall)
    print(cm)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=classname)
    cm_display.plot(cmap=plt.cm.Blues, colorbar=False)
    plt.savefig(cstr.datarootdir+'confmat.eps', bbox_inches='tight')
    plt.savefig(cstr.datarootdir+'confmat.pdf', bbox_inches='tight')

    #return

    #print('Executing main() ....')
    #model = CNN(20, 4, 5, 6, 7)  #  DEBUG ; return
    #params = model.get_params()
    #print('params=', params)

    #key_value_pair = {'activation': 'sigmoid'}
    #model.set_params(**key_value_pair)

    #params = model.get_params()
    #print('params=', params)
    #keys = model.get_params().keys()
    #print('keys=', keys)

    ## model.build(input_shape=(model.dim, model.chan))


def PCA():
    from CSTR_plot import read_X, dataframe2sklearn
    #  _ = run_experiment(exp_normal)
    # from CSTR_machine_learning import sklearn_apply
    exp_PCA = {'id': None, 'theta': 1, 'randseed': 1234, 'fortran_rand': False,
               'timehoriz': 500, 'plotmask': None, 'faults': (
                      Fault(id=2, EXTENT0=200, DELAY=400, TC=0.1), )}
    cstr = run_experiment(exp_PCA, do_run=True, do_plot=False)
    datafn = cstr.datafn
    print('Opening input data file %s.' % datafn)
    df = read_X(datafn=datafn)
    X, y, featname = dataframe2sklearn(df)
    labels = y
    # print('labels =\n', labels); input('...')
    plot_PCA(cstr, X, y, plot_time_axis=True,
            dropfigfile=cstr.datarootdir+'PCA', title='First 2 PC with time axis ')


def paper():
    #PCA()
    #visualize()
    fault_detect()
    #fault_diagnose(standardize=True, do_run=True, do_tSNE=True)
    #fault_diagnose(standardize=False, do_run=True, do_tSNE=True)


if __name__ == "__main__":
    paper()
'''
Fault        Variable      Nominal value
 1            NO FAULT
 2            R1                100          BLOCKAGE AT TANK OUTLET
 3            R9                  0          BLOCKAGE IN JACKET
 4            R8            1000000          JACKET LEAK TO ENVIRONMENT
 5            R7            1000000          JACKET LEAK TO TANK
 6            R2            1000000          LEAK FROM PUMP
 7            PP              48000          LOSS OF PUMP PRESSURE
 8            UA               1901          JACKET EXCHANGE SURFACE FOULING
 9            QEXT                0          EXTERNAL HEAT SOURCE (SINK)
10            BETA1           25000          PRIMARY REACTION ACTIVATION ENERGY
11            BETA2           45000          SECONDARY REACTION ACTIVATION ENERGY
12            FLOW1            0.25          ABNORMAL FEED FLOWRATE
13            T1                 30          ABNORMAL FEED TEMPERATURE
14            CA0                20          ABNORMAL FEED CONCENTRATION
15            T3                 20          ABNORMAL COOLING WATER TEMPERATURE
16            PCW             56250          ABNORMAL COOLING WATER PRESSURE
17            JEP                 0          ABNORMAL JACKET EFFLUENT PRESSURE
18            REP                 0          ABNORMAL REACTOR EFFLUENT PRESSURE
19            SP1                 2          ABNORMAL LEVEL CONTROLLER SETPOINT
20            SP2                80          ABNORMAL TEMPERATURE CONTROLLER SETPOINT
21            100-V1           25.3          CONTROL VALVE (CV-1) STUCK
22            100-V2           40.7          CONTROL VALVE (CV-2) STUCK
Sensor Fault
 1            MEAS1            20            FEED_CONCENTRATION_SENSOR
 2            MEAS2          0.25            FEED_FLOWRATE_SENSOR
 3            MEAS3            30            FEED_TEMPERATURE_SENSOR
 4            MEAS4             2            REACTOR_LEVEL_SENSOR
 5            MEAS5          2.85            CONCENTRATION_A_SENSOR
 6            MEAS6         17.11            CONCENTRATION_B_SENSOR
 7            MEAS7            80            REACTOR_TEMPERATURE_SENSOR
 8            MEAS8           0.9            COOLING_WATER_FLOWRATE_SENSOR
 9            MEAS9          0.25            PRODUCT_FLOWRATE_SENSOR
10            MEAS10           20            COOLING_WATER_TEMPERATURE_SENSOR
11            MEAS11        56250            COOLING_WATER_PRESSURE_SENSOR
12            MEAS12         25.3            LEVEL_CONTROLLER_OUTPUT_SIGNAL
13            MEAS13         40.7            CW_FLOW_CONTROLLER_OUTPUT_SIGNAL
14            MEAS14          0.9            COOLING_WATER_SETPOINT

'''
