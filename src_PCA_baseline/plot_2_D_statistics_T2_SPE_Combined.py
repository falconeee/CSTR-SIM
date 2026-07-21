import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def SPE(x, Ctil):
    return np.dot(np.dot(x.T, Ctil), x)

def T2(x, D):
    return np.dot(np.dot(x.T, D), x)

def combined_phi(x, PHI):
    return np.dot(np.dot(x.T, PHI), x)


def plot_2_D_statistics(training):
    from ellipse import plot_ellipse
    '''
        Residual space has no more covariance and variances are eigenvalues
        of covariance matrix which are identical to the variances of the
        covariance matrix
        and are also identical to the eigenvalues of the original data
        associated to the residual space = Lambda[n-l:]
        command to verify: print(np.cov(Xresidual, rowvar=False))
        ==>
        covariance matrix is unit matrix
        
    '''

    # Angle and its cosine between two vectors
    def angle_between(u,v):
        cosangle = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v) # -> cosine of the angle
        angle = np.arccos(np.clip(cosangle, -1, 1)) # angle in radians
        return angle, cosangle
    
    X = training.get('Xtrain')
    m, n = X.shape # m samples, n variables, data matrix has dimension
    l = training.get('estimated_num_principal_components')
    Lambda = training.get('Lambda')
    Ltil = np.diag(Lambda[l:])
    Ptil = training.get('Ptil')
    Ctil = training.get('M_c_SPE')
    deltasqr = training.get('deltasqr')
    
    plot_subspace = True
    plot_subspace = False

    plot_SPE = n-l == 2
    print('Number of principal components = ', l, 'plot_SPE=', plot_SPE)
    print('m=', m, 'n=', n); #input('...')

    if plot_SPE:
        print('M=Ptil=\n', Ptil, 'shape=', Ptil.shape); input('...')
        Xresidual = np.dot(X, Ptil)
        #print('Xresidual=\n', Xresidual); input('...')
        if plot_subspace:
            plt.scatter(Xresidual[:, 0], Xresidual[:, 1], s=10,
                        facecolors='none', edgecolors='green', linewidths=0.5)
        
        plt.axis('equal')
        plt.title('SPE')

        '''
        plot_kwargs = {'color':'g','linestyle':'-','linewidth':1,'alpha':0.2}
        fill_kwargs = {'color':'g','alpha':0.1}
        a = b = np.sqrt(deltasqr); angle = 0.0
        plot_ellipse(semimaj=a,semimin=b,phi=angle,x_cent=0.0,y_cent=0.0,
                    theta_num=1000,ax=plt.gca(),plot_kwargs=plot_kwargs,
                    fill=True,fill_kwargs=fill_kwargs,data_out=False,
                    cov=None,mass_level=0.68)
        '''
        normalSPE = np.zeros(m)
        for i in range(m):
            x = X[i]
            spe = SPE(x, Ctil)
            normalSPE[i] = spe <= deltasqr
            # Alternative
            xres = Xresidual[i];
            normalSPE[i] = LA.norm(np.dot(xres, Ptil.T)) < np.sqrt(deltasqr)
            #print('Normal=', normal, 'Normal res=', normalres, 'i=',i,'x=',x,'z=',z,'SPE=',spe)
            if not normalSPE[i] and plot_subspace:
                plt.scatter(Xresidual[i, 0], Xresidual[i, 1], s=10, color='red')
    
        newaxes = np.dot(np.identity(n), Ptil)
        angle, cosangle = angle_between(newaxes[0], newaxes[1])
        #angle, cosangle = angle_between(newaxes[0,:].T, newaxes[1,:].T) # gives the same for n=4,l=2
        sinangle = np.sin(angle)
        print('angle between largest eigenvector and x-axis=', angle, '=', angle/np.pi, 'pi')
        RotMat = np.array([[cosangle, -sinangle], [sinangle, cosangle]])
        
        limit = delta = np.sqrt(deltasqr)
        #covarres = np.cov(Xresidual, rowvar=False)  # = Ltil
        
        a = 1.0/np.sqrt(Ltil[0,0])
        b = 1.0/np.sqrt(Ltil[1,1])
        plot_kwargs = {'color':'r', 'linestyle':'-', 'linewidth':1, 'alpha':0.1}
        fill_kwargs = {'color':'r', 'alpha':0.1}
        plot_ellipse(semimaj=limit*a, semimin=limit*b, phi=-angle, x_cent=0.0, y_cent=0.0,
                     theta_num=1000, ax=plt.gca(), plot_kwargs=plot_kwargs,
                     fill=True, fill_kwargs=fill_kwargs, data_out=False,
                     cov=None, mass_level=0.68)
        
        #print('RotMat=', RotMat)
        ScaleMat = np.array([[a, 0.0], [0.0, b]])
        
        #Xback = np.dot(np.dot(Xresidual, RotMat), ScaleMat)
        Xback = np.dot(np.dot(Xresidual, ScaleMat), RotMat)
        #Xback = np.dot(Xresidual,ScaleMat)
        #print('Xback=\n', Xback)
        backwardincol = 'g'
        backwardoutcol = 'r'
        for i in range(m):
            edgecolors = backwardoutcol if not normalSPE[i] else backwardincol
            plt.scatter(Xback[i, 0], Xback[i, 1], s=10,
                        facecolors='none', edgecolors=edgecolors, linewidths=0.5)
        
        #plt.scatter(Xback[:, 0], Xback[:, 1], s=10,
                    #facecolors='none', edgecolors=backwardincol, linewidths=0.5)
        #for i in range(m):
            #if not normalSPE[i]:
                #plt.scatter(Xback[i, 0], Xback[i, 1], s=10, color=backwardoutcol)
        plt.show()
        #plt.legend()
        #return
        
    plot_T2 = l == 2
    if plot_T2:
        '''
            T2
        '''
        #plt.clf()
        plt.figure(2)
        plt.title('T2')
        M_c_T2 = training.get('M_c_T2')
        tau2 = training.get('tau2')
        P = training.get('P')
        L = training.get('L')
        M_T2 = np.dot(P, np.diag(1.0/np.sqrt(Lambda[:l])) )
        X_PC = np.dot(X, M_T2)
        #print('X_PC=\n', X_PC) 
        if plot_subspace:
            plt.scatter(X_PC[:, 0], X_PC[:, 1], s=10,
                        facecolors='none', edgecolors='g', linewidths=0.5)
            plt.axis('equal')
            
            plot_kwargs = {'color':'g','linestyle':'-','linewidth':1,'alpha':0.2}
            fill_kwargs = {'color':'g','alpha':0.1}
            a = b = np.sqrt(tau2); angle = 0.0
            plot_ellipse(semimaj=a,semimin=b,phi=angle,x_cent=0.0,y_cent=0.0,
                        theta_num=1000,ax=plt.gca(),plot_kwargs=plot_kwargs,
                        fill=True,fill_kwargs=fill_kwargs,data_out=False,
                        cov=None,mass_level=0.68)
        normalT2 = np.zeros(m)
        for i in range(m):
            x = X[i]
            t2 = T2(x, M_c_T2)
            normalT2[i] = t2 <= tau2
            # Alternative
            xpc = X_PC[i];
            normalT2[i] = LA.norm(np.dot(xpc, P.T)) < np.sqrt(tau2)
            if not normalT2[i] and plot_subspace:
                plt.scatter(X_PC[i, 0], X_PC[i, 1], s=10, color='g')
        
    
        newaxes = np.dot(np.identity(n), P)
        angle, cosangle = angle_between(newaxes[0], newaxes[1])
        #angle, cosangle = angle_between(newaxes[0,:].T, newaxes[1,:].T) # gives the same for n=4,l=2
        sinangle = np.sin(angle)
        print('angle between largest eigenvector and x-axis=', angle, '=', angle/np.pi, 'pi')
        RotMat = np.array([[cosangle, -sinangle], [sinangle, cosangle]])
        
        limit = tau = np.sqrt(tau2)
        
        a = 1.0/np.sqrt(L[0,0])
        b = 1.0/np.sqrt(L[1,1])
        plot_kwargs = {'color':'r', 'linestyle':'-', 'linewidth':1, 'alpha':0.1}
        fill_kwargs = {'color':'r', 'alpha':0.1}
        plot_ellipse(semimaj=limit*a, semimin=limit*b, phi=-angle, x_cent=0.0, y_cent=0.0,
                     theta_num=1000, ax=plt.gca(), plot_kwargs=plot_kwargs,
                     fill=True, fill_kwargs=fill_kwargs, data_out=False,
                     cov=None, mass_level=0.68)
        
        #print('RotMat=', RotMat)
        ScaleMat = np.array([[a, 0.0], [0.0, b]])
        
        #Xback = np.dot(np.dot(Xresidual, RotMat), ScaleMat)
        Xback = np.dot(np.dot(X_PC, ScaleMat), RotMat)
        #Xback = np.dot(Xresidual,ScaleMat)
        #print('Xback=\n', Xback)
        backwardincol = 'g'
        backwardoutcol = 'r'
        #plt.scatter(Xback[:, 0], Xback[:, 1], s=10,
                    #facecolors='none', edgecolors=backwardincol, linewidths=0.5)
        for i in range(m):
            edgecolors = backwardoutcol if not normalT2[i] else backwardincol
            plt.scatter(Xback[i, 0], Xback[i, 1], s=10,
                            facecolors='none', edgecolors=edgecolors, linewidths=0.5)
            #if not normalT2[i]:
                #plt.scatter(Xback[i, 0], Xback[i, 1], s=10, facecolors='none', color=backwardoutcol)
            #else:
                #plt.scatter(Xback[i, 0], Xback[i, 1], s=10,
                            #facecolors='none', edgecolors=backwardincol, linewidths=0.5)
        plt.show()

    #'''
        #Combined
        #On the contrary to SPE (dimension l) and T2 (dimension n-l),
        #the dimension of the projection space of the combined index is n,
        #hence the dimension for visualization of the projected samples is n
    #'''
    #M_c_combined = PHI = training.get('M_c_combined')
    #zeta2 = training.get('zeta2')


    #plt.figure(3)
    #X_combined = np.dot(X, M_c_combined)
    ##print('X_combined=\n', X_combined)      
    #plt.scatter(X_combined[:, 0], X_combined[:, 1], s=10,
                #facecolors='none', edgecolors='g', linewidths=0.5)
    #plt.axis('equal')
    
    #'''
    #plot_kwargs = {'color':'g','linestyle':'-','linewidth':1,'alpha':0.2}
    #fill_kwargs = {'color':'g','alpha':0.1}
    #a = b = np.sqrt(zeta2); angle = 0.0
    #plot_ellipse(semimaj=a,semimin=b,phi=angle,x_cent=0.0,y_cent=0.0,
                 #theta_num=1e3,ax=plt.gca(),plot_kwargs=plot_kwargs,
                 #fill=True,fill_kwargs=fill_kwargs,data_out=False,
                 #cov=None,mass_level=0.68)
    #'''
    #normal_combined = np.zeros(m)
    #y = np.zeros(m,dtype=np.int)
    
    #combined_phi_X = np.sum( np.dot(X, PHI) * X, axis=1) # Eq. (4)
    ##print('combined_phi_X=', combined_phi_X)
    #NormalConditionCombined = combined_phi_X < zeta2
    #numFaultsCombined = list(NormalConditionCombined).count(False)

    
    #if numFaultsCombined == 0:
        #classname=np.array(['Normal'], dtype='|S10')
        #numclasses=1
    #else:
        #classname=np.array(['Normal', 'Fault'], dtype='|S10')
        #numclasses=2
    
    #for i in range(m):
        #x = X[i]
        #phi = combined_phi(x, PHI)
        #normal_combined[i] = phi <= zeta2
        ## Alternative
        ##xc = X_combined[i];
        ##normal_combined[i] = LA.norm(np.dot(xc, PHI.T)) < np.sqrt(zeta2)
        #if not normal_combined[i]:
            #y[i] = 1
            ##print('i=', i, 'phi=', phi, 'X_combined=', X_combined[i])
            ##plt.scatter(X_combined[i, 0], X_combined[i, 1], s=10, color='g')
    
    
    from sklearn.manifold import TSNE
    from mpl_toolkits.mplot3d import Axes3D
    
    from scatter import scatter
    def plot_tSNE(X, y, classname, numclasses=1, n_components=2):
        print ('Generating tSNEplot...')
        #print ('plot_tSNE> classname=', classname, 'numclasses=', numclasses)
        X_embedded = TSNE(n_components=n_components).fit_transform(X)
        Xplot = X_embedded
        xlab = 'tSNE Embedded dim 1'
        ylab = 'tSNE Embedded dim 2'
        tit = 'tSNE Plot'
        #print('Xplot.shape=', Xplot.shape, 'y.shape=', y.shape, 'type(y)=', type(y),
        #      'Xplot.size=', Xplot.size, 'y.size=', y.size)
        cmap = plt.get_cmap('gnuplot')
        color = [cmap(i) for i in np.linspace(0, 1, numclasses)]
        if n_components==2:
            #plt.scatter(Xplot[:, 0], Xplot[:, 1])
            ax = plt.gca()
            ax.set_title(tit)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            #ax.scatter(Xplot, y, classname, numclasses, title=tit, xlabel=xlab, ylabel=ylab)
            for i in range(numclasses):
                idx = np.where(y == i)
                ax.scatter(Xplot[idx, 0], Xplot[idx, 1], c=color[i], label=classname[i]) 
        if n_components==3:
            zlab = 'tSNE Embedded dim 3'
            fig = plt.figure(1, figsize=(8, 6))
            
            ax = Axes3D(fig, elev=-150, azim=110)
            #ax.w_yaxis.set_ticklabels([])
            ax.set_title(tit)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_zlabel(zlab)
            #ax.w_zaxis.set_ticklabels([])
            for i in range(numclasses):
                idx = np.where(y == i)
                ax.scatter(Xplot[idx, 0], Xplot[idx, 1], Xplot[idx, 2],
                           c=color[i], label=classname[i]) 
        plt.legend(loc=2)
        plt.show()
    
    #plt.figure(4)
    #plot_tSNE(X_combined, y, classname, numclasses, n_components=2)
    plt.show()

