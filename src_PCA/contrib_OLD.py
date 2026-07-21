def contribution_plot_OLD(training, fault_data_window, discard_first_test=None, fault_start=None, fault_stop=None,
        fault_detection_index='SPE', semilogy=False, benchmark=None, dropfigdir=None, figext='.eps'):

    print('contribution_plot> fault_data_window=', fault_data_window)
    numvar = training.get('num_variables')

    training_data = training.get('Xtrain_norm')
    #fault_data = training.get('Xtest_norm')

    #training_data = training.get('Xtrain')
    #fault_data = training.get('Xtest')

    M_sqrt, control_limit, M = get_mat_and_limit(training, fault_detection_index)


    if benchmark == 'pubBnB':   # Special case: use the explicit fault pattern
       fault_data = get_pubBnB_fault_data()
    
    if fault_detection_index == 'M2':
        meanX = training.get('meanX')
        #fprintf('M2 index: Special non-standardized training and test data...')
        training_data = training.get('Xtrain') - meanX
        fault_data = training.get('Xtest') - meanX

    Index_train, Index_per_variable_train = Index_and_individual_contribs(training_data, M_sqrt)
    #print('Index_train=', Index_train, 'Index_train.shape=', Index_train.shape);
    Index_train = abs(Index_train)  # might get some complex values
    Index_per_variable_train = abs(Index_per_variable_train)

    fault_data_all = fault_data_window

    Index_test_all, Index_per_variable_test_all = Index_and_individual_contribs(fault_data_all, M_sqrt, verbose=False)
    Index_test_all = abs(Index_test_all)  # might get some complex values
    Index_per_variable_test_all = abs(Index_per_variable_test_all)

    startidx = fault_start-1
    stopidx = fault_stop-1

    Index_test, Index_per_variable_test = Index_and_individual_contribs(fault_data_window, M_sqrt, verbose=False)
    Index_test = abs(Index_test)  # might get some complex values
    Index_per_variable_test = abs(Index_per_variable_test)
    print('Contribution plot> Index_test=', Index_test); quit()

    # C o n t r i b u t i o n  p l o t 
    #cplot, ax = plt.subplots(); # Create a figure and a set of subplots
    cplot, ax = plt.gcf(), plt.gca()    # Current figure and axes

    # explicitly choose the sample for which the individual contributions are to be plotted
    #fprintf('individual_contribs=', individual_contribs)

    # [2], Fig.2
    tex_setup(usetex=usetex)
    
    #print('Contribution plot from individual index from ', startidx,
    #        'to', stopidx, 'of', fault_data_window.shape[0], 'samples')

    individual_contribs = Index_per_variable_test.mean(axis=0)

    height = individual_contribs
    y_pos = np.arange(numvar)
    plt.bar(y_pos, height, color='#DDD0FF', label=None)
    ax = plt.gca()
    for i, v in enumerate(height):
        ax.text(i, height[i], '{:.2f}'.format(height[i]), fontsize=7, verticalalignment='bottom',
            horizontalalignment='center', alpha=1.0, rotation=0, color='black')

    plt.xticks(y_pos, 1+y_pos, rotation=90)

    barnames = training.get('featname')
    if not barnames is None:
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, ymax])   # Seems to be necessary to plot correctly
        yrange = ax.get_ylim()
        yoffset = 0.02*yrange[1]    # percentage of extension of y-axis
        plt.tick_params(labelbottom=False)#, bottom=False)
        #fprintf('yrange=', yrange, 'yoffset=', yoffset);
        for i, v in enumerate(height):
            ax.text(i, yoffset, barnames[i], fontsize=7, verticalalignment='bottom',
                    horizontalalignment='center', alpha=0.6, rotation=90, color='black')

    plt.xlabel('variable')
    plt.title('Individual Contribution Plot for Index ' + fault_detection_index)
    plt.ylabel('contribution')
    limit_label = fault_detection_index + ' control limit %.2f' % control_limit
    #if not control_limit is None:
    #    plt.axhline(y=control_limit, color='r', linestyle='--', label=limit_label)
    #    plt.legend()
    if not dropfigdir is None:
        # assemble file name of contribution plot with time stamp
        nowstr = datetime #.now().strftime('%Y_%m_%d__%H_%M_%S')
        fname = dropfigdir + nowstr + '_contrib' + figext
        #fname = '/home/thomas/Dropbox/papers/2019_tmp/figs/test1.eps'    # DEBUG
        fprintf('Saving contribution plot in ', fname)
        print('Saving contribution plot in ', fname)
        plt.savefig(fname, dpi=1200)
    plt.show()


    # T i m e  E v o l u t i o n
    timeevolution, ax = plt.subplots(); # Create a figure and a set of subplots
    tex_setup(usetex=usetex)

    widthcm = 8 # Real sizes later in the LaTeX file
    heigthcm = 6
    timeevolution.set_size_inches([cm2inch(widthcm), cm2inch(heigthcm)])
    #timeevolution.tight_layout(pad=0, h_pad=0, w_pad=0)

    plt.title('Temporal Evolution of ' + fault_detection_index + ' Index')
    plt.xlabel('sample')
    plt.ylabel( fault_detection_index + ' Index')
    if control_limit != None:
        limit_label = 'Control limit'
        plt.axhline(y=control_limit, color='r', linestyle='--', label=limit_label, linewidth=0.75)
    numpoints_train = Index_train.shape[0]

    # End of training data
    plt.axvline(x=numpoints_train-1, color='cyan', linestyle='-', linewidth=0.5, label='End of training')
    if not discard_first_test is None:
        # Start of used test data for detection
        plt.axvline(x=numpoints_train-1+discard_first_test, color='magenta', linestyle='-', linewidth=0.5, label='Fault triggered')

    numpoints_test = Index_test_all.shape[0]
    xpostrain = np.linspace(0, numpoints_train-1, num=numpoints_train)
    xpostest = np.linspace(numpoints_train-1, numpoints_train+numpoints_test-1, num=numpoints_test)
    fprintf('Contribution plot: numpoints_train=', numpoints_train, 'numpoints_test=', numpoints_test)
    plotfunc = plt.plot
    if semilogy:
        plotfunc = plt.semilogy
    plotfunc(xpostrain, Index_train, linewidth=0.25, color='green', label='Normal')
    plotfunc(xpostest, Index_test_all, linewidth=0.25, color='blue', label='Test')

    # change the xticks of the fault data set
    xticks = ax.get_xticks().tolist()
    #print('xticks=', xticks)
    for i in range(len(xticks)):
        t = xticks[i]
        #print('t=', t)
        if t > numpoints_train:
            t -= numpoints_train
            #print('over new=', t)
        xticks[i] = int(t)
    ax.set_xticklabels(xticks)

    if not fault_stop is None:
        intvalcol = 'grey'
        intvalcol = '0.7'
        # Plot the interval limits of the test data used for the calculus of contribution
        #plt.axvline(x=numpoints_train-1+fault_start, color='cyan', linestyle='-', linewidth=0.75, label='Fault detection window start')
        #plt.axvline(x=numpoints_train-1+fault_stop, color='magenta', linestyle='-', linewidth=0.75, label='Fault detection window stop')

        startpos = numpoints_train-1+fault_start
        stoppos = numpoints_train+fault_stop
        #if not discard_first_test is None:
        #    startpos += discard_first_test
        #    stoppos += discard_first_test

        intval = np.arange(startpos, stoppos)
        #ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, ymax])   # Seems to be necessary to plot correctly
        ax.fill_between(intval, ymin, ymax, color=intvalcol, alpha=0.3, label='Fault detection\nwindow')


    legend_loc = None
    legend_loc = 'best'
    legend_loc ='lower center'
    legend_loc ='upper left'
    #ax.legend(bbox_to_anchor=(0.0, 1.25))
    #plt.legend(loc=legend_loc, ncol=2)
    #plt.legend(loc='upper right', bbox_to_anchor=(0.80, 1.05),  ncol=2)
    plt.legend(loc='lower right', bbox_to_anchor=(0.80, 0.05),  ncol=2)
    if not dropfigdir is None:
        # assemble file name of time evolution plot with time stamp
        fname = dropfigdir + nowstr + '_evolution' + figext
        #fname = '/home/thomas/Dropbox/papers/2019_tmp/figs/test2.eps'    # DEBUG
        fprintf('Saving time evolution plot in ', fname)
        plt.savefig(fname, dpi=1200)
    plt.show()

