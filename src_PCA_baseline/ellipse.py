import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
#import pylab as mp

# http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix
# https://casper.berkeley.edu/astrobaki/index.php/Plotting_Ellipses_in_Python

def plot_ellipse(semimaj=1, semimin=1, phi=0, x_cent=0, y_cent=0, theta_num=1000,
                 ax=None, plot_kwargs=None,
                 fill=False, fill_kwargs=None, data_out=False,
                 cov=None, mass_level=0.68):
    '''
        An easy to use function for plotting ellipses in Python 2.7!

        The function creates a 2D ellipse in polar coordinates then transforms to cartesian coordinates.
        It can take a covariance matrix and plot contours from it.
        
        semimaj : float
            length of semimajor axis (always taken to be some phi (-90<phi<90 deg) from positive x-axis!)

        semimin : float
            length of semiminor axis

        phi : float
            angle in radians of semimajor axis above positive x axis

        x_cent : float
            X coordinate center

        y_cent : float
            Y coordinate center

        theta_num : int
            Number of points to sample along ellipse from 0-2pi

        ax : matplotlib axis property
            A pre-created matplotlib axis

        plot_kwargs : dictionary
            matplotlib.plot() keyword arguments

        fill : bool
            A flag to fill the inside of the ellipse 

        fill_kwargs : dictionary
            Keyword arguments for matplotlib.fill()

        data_out : bool
            A flag to return the ellipse samples without plotting

        cov : ndarray of shape (2,2)
            A 2x2 covariance matrix, if given this will overwrite semimaj, semimin and phi

        mass_level : float
            if supplied cov, mass_level is the contour defining fractional probability mass enclosed
            for example: mass_level = 0.68 is the standard 68% mass

    '''
    # Get Ellipse Properties from cov matrix
    if cov is not None:
        eig_vec,eig_val,u = np.linalg.svd(cov)
        # Make sure 0th eigenvector has positive x-coordinate
        if eig_vec[0][0] < 0:
            eig_vec[0] *= -1
        semimaj = np.sqrt(eig_val[0])
        semimin = np.sqrt(eig_val[1])
        if mass_level is None:
            multiplier = np.sqrt(2.279)
        else:
            distances = np.linspace(0,20,20001)
            chi2_cdf = chi2.cdf(distances,df=2)
            multiplier = np.sqrt(distances[np.where(np.abs(chi2_cdf-mass_level)==np.abs(chi2_cdf-mass_level).min())[0][0]])
        semimaj *= multiplier
        semimin *= multiplier
        phi = np.arccos(np.dot(eig_vec[0],np.array([1,0])))
        if eig_vec[0][1] < 0 and phi > 0:
            phi *= -1

    # Generate data for ellipse structure
    theta = np.linspace(0,2*np.pi,theta_num)
    r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    data = np.array([x,y])
    S = np.array([[semimaj,0],[0,semimin]])
    R = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
    T = np.dot(R,S)
    data = np.dot(T,data)
    data[0] += x_cent
    data[1] += y_cent

    # Output data?
    if data_out == True:
        return data

    # Plot!
    return_fig = False
    if ax is None:
        return_fig = True
        fig,ax = plt.subplots()

    if plot_kwargs is None:
        ax.plot(data[0],data[1],color='b',linestyle='-')
    else:
        ax.plot(data[0],data[1],**plot_kwargs)

    if fill == True:
        ax.fill(data[0],data[1],**fill_kwargs)

    if return_fig == True:
        return fig



def main():

    plot_ellipse(semimaj=5, semimin=1, phi=-0.396907 )#np.pi/3)
    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlabel('x axis', fontsize=20)
    ax.set_ylabel('y axis', fontsize=20)
    ax.grid(True)
    
    plot_kwargs = {'color':'r', 'linestyle':'-', 'linewidth':3, 'alpha':0.5}
    fill_kwargs = {'color':'r', 'alpha':0.3}
    plot_ellipse(semimaj=5, semimin=2, phi=np.pi/2.5, x_cent=20, y_cent=40,
                 ax=ax, plot_kwargs=plot_kwargs, fill=True, fill_kwargs=fill_kwargs)

    plt.show()
    
    plot_kwargs = {'color':'b', 'linestyle':'-', 'linewidth':3, 'alpha':0.8}
    fill_kwargs = {'color':'b', 'alpha':0.7}
    #plot_ellipse(semimaj=2.5,semimin=1, phi=np.pi/2.5, x_cent=20, y_cent=40,
    #             ax=ax, plot_kwargs=plot_kwargs, fill=True, fill_kwargs=fill_kwargs)
    
    plt.show()
    #print('Show 3')
    #input('Press <ENTER> to exit ...')

if __name__ == '__main__':
    main()
