# Finite Horizon Dynamic Programming problem
# featuring labor and savings choice
# as well as exogenous time-variant wage and taxation
# Coded by Jan Ertl, Oct 1, 2017

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def finitevfi_wlabor(gamma = 2, beta = 0.97, r = 0.03, psi = 0.1,
                     T = 50, b = 0.5, tau = 0,
                     amin = 0, amax = 10, na = 501, ainitial = 5, nh = 2):
    """Finite Horizon Dynamic Programming problem featuring labor and savings choice
    as well as exogenous time-variant wage and taxation
    Inputs:

    """
    ## Initializing

    # period return

    R = 1 + r


    # asset and labor grids

    a_grid = np.linspace(amin,amax,na)
    h_grid = np.linspace(0,1,nh) # max labor is normalized to 1


    # utility function

    def u(c,h):
        return (c**(1-gamma)-1)/(1-gamma) - psi*h


    # wages with tax tau

    def w(t,tau=0):
        if t > T/2:
            return (1-tau)*(T + 1 - t)/10
        else:
            return (1-tau)*t/10

    T = T + 1

    # initialize output matrices and indices
    V = np.zeros((na,T)) # value function
    sav = np.zeros((na,T)) # savings
    savind = np.zeros((na,T)) # index of savings choice based on current assets
    con = np.zeros((na,T)) # consumption
    h = np.zeros((na,T)) # labor supply


    ## Decisions at t=T


    hind = np.zeros((na,T))
    conh = np.zeros((na,nh))
    Vh = np.zeros((na,nh))

    # the agent does not have a savings motive in the final period
    sav[:,T-1] =0

    for i in range(nh):
        conh[:,i] = R*a_grid + w(T)*h_grid[i] + (1-h_grid[i])*b - sav[:,T-1]
        Vh[:,i] = u(conh[:,i],h_grid[i])

    V[:,T-1] = np.amax(Vh,1)
    hind = np.argmax(Vh,1)

    con[:,T-1] = np.choose(hind, conh.T)

    h[:,T-1] = np.choose(hind, h_grid.T)



    cash = np.zeros(conh.shape)

    ## Solving backward
    print('Solving backward')
    for it in np.arange(T-2,-1,-1):
        #print("Solving at age ", (it+1))
        for ia in range(0,na):
            for i in range(nh):
                cash[:,i] = R*a_grid[ia] + w(it,tau)*h_grid[i] + (1-h_grid[i])*b
                conh[:,i] = np.maximum((cash[:,i] - a_grid), 1 * 10 ** -10)
                Vh[:,i] = u(conh[:,i],h_grid[i]) + beta * V[:,it+1]

            # given each possible labor choice, select best savings/consumption choice
            hind = np.argmax(Vh,1)
            cashchoice = np.choose(hind, cash.T)
            conchoice = np.choose(hind, conh.T)
            hchoice = np.choose(hind, h_grid.T)

            # choose the labor supply that maximizes the value function

            Vchoice = np.amax(Vh,1)
            Vind = np.argmax(Vchoice)

            # index and respective policies
            savind[ia,it] = Vind
            V[ia,it] = np.max(Vchoice)
            h[ia,it] = hchoice[Vind]
            sav[ia,it]= a_grid[Vind]
            con[ia,it] = cashchoice[Vind] - sav[ia,it]


    ## Simulate
    aindsim = np.zeros(T+1)
    hsim = np.zeros(T+1)
    csim = np.zeros(T+1)
    asim = np.zeros(T+1)

    # initial assets
    # ainitial

    # allocate to nearest point on agrid
    inter = interpolate.interp1d(a_grid,range(0,na),'nearest')

    aindsim[0] = inter(ainitial) #a_grid[np.int(inter(ainitial))]
    asim[0] = a_grid[np.int(aindsim[0])]

    print('Simulating')
    for it in range(0,T):
        #print('Simulating, time period', (it+1))
        #asset choice
        aindsim[it+1] = savind[np.int_(aindsim[it]),it]
        asim[it+1] = a_grid[np.int(aindsim[it+1])]
        hsim[it] = h[np.int(aindsim[it]),it]
        csim[it] = R*asim[it] + w(it,tau)*hsim[it] + (1 - hsim[it])*b - asim[it+1]

    #hsim[T] = h[np.int(aindsim[T-1]),T-1]
    #csim[T] = R*asim[T] + w(T,tau)*hsim[T] + (1 - hsim[T])*b

    ## Plot

    wages = [w(t,tau) for t in range(T+1)]

    fig_1 = plt.figure(figsize = (20,10))

    #consumption and income path
    plt.title('Consumption, Wealth, and Labor')

    plt.subplot(1,3,1)
    plt.title('Consumption path')
    plt.plot(range(0,T+1),wages, 'k-', lw=1)
    #plt.plot(range(1,T+1),csim[0:T],'r--', lw=1)
    plt.plot(range(0,T),csim[0:T],'r--', lw=1)
    plt.grid
    plt.title('Consumption path')

    # Wealth path
    plt.subplot(1,3,2)
    plt.title('Wealth Path')
    plt.plot(range(0,T+1),asim, 'b-',lw=1)


    # Labor path
    plt.subplot(1,3,3)
    plt.title('Labor supply')
    plt.plot(range(0,T+1),hsim, 'b-',lw=1)
    plt.show()


    return 0



    
