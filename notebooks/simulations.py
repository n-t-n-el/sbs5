# adapted from https://biocircuits.github.io/technical_appendices/16b_profiling_for_speed.html
import numpy as np
import numba
import matplotlib.pyplot as plt
from scipy.stats import gamma as sgamma

@numba.njit
def sample_discrete_numba(probs, probs_sum):
    q = np.random.rand() * probs_sum

    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1

    return i - 1


@numba.njit
def prop_func(
    propensities, population, t, parameters
):
    """Updates an array of propensities given a set of parameters
    and an array of populations.
    """
    # Unpack
    x,y,z = population
    u,b,r1,r2,n,epsilon = parameters
    # Update propensities
    propensities[0] = u  # Damage
    propensities[1] = r1*x*z# Detection
    propensities[2] = r2*(n-z)*(1-epsilon)  # Correct repair
    propensities[3] = r2*(n-z)*epsilon  # Incorrect repair

@numba.njit
def gillespie_draw_numba(propensities, population, t, parameters):
    """
    Draws a reaction and the time it took to do that reaction.

    Assumes that there is a globally scoped function
    `prop_func` that is Numba'd with nopython=True.
    """
    # Compute propensities
    prop_func(propensities, population, t, parameters)

    # Sum of propensities
    props_sum = np.sum(propensities)

    # Compute time
    time = np.random.exponential(1 / props_sum)

    # Draw reaction given propensities
    rxn = sample_discrete_numba(propensities, props_sum)

    return rxn, time


@numba.njit
def gillespie_ssa_numba(update, population_0, time_points, parameters):
    """
    Uses the Gillespie stochastic simulation algorithm to sample
    from proability distribution of particle counts over time.

    Parameters
    ----------
    update : ndarray, shape (num_reactions, num_chemical_species)
        Entry i, j gives the change in particle counts of species j
        for chemical reaction i.
    population_0 : array_like, shape (num_chemical_species)
        Array of initial populations of all chemical species.
    time_points : array_like, shape (num_time_points,)
        Array of points in time for which to sample the probability
        distribution.
    parameters : tuple, default ()
        The set of parameters to be passed to propensity_func.

    Returns
    -------
    sample : ndarray, shape (num_time_points, num_chemical_species)
        Entry i, j is the count of chemical species j at time
        time_points[i].

    Notes
    -----
    .. Assumes that there is a globally scoped function
       `propensity_func` that is Numba'd with nopython=True.
    """
    # Initialize output
    pop_out = np.empty((len(time_points), update.shape[1]), dtype=np.int64)

    # Initialize and perform simulation
    i_time = 1
    i = 0
    i_burst = 0
    u,b,r1,r2,n,epsilon = parameters
    size = int(time_points[-1]/u)*100
    burst_size = np.random.geometric(p=1/b,size=size)
    t = time_points[0]
    population = population_0.copy()
    pop_out[0,:] = population
    propensities = np.zeros(update.shape[0])
    while i < len(time_points):
        while t < time_points[i_time]:
            # draw the event and time step
            event, dt = gillespie_draw_numba(propensities, population, t, parameters)

            # Update the population
            population_previous = population.copy()
            up = update[event,:]
            if up[0] == 1: 
                population[0] += burst_size[i_burst % size] # burst
                i_burst += 1
            else: population += up

            # Increment time
            t += dt

        # Update the index (Have to be careful about types for Numba)
        i = np.searchsorted((time_points > t).astype(np.int64), 1)

        # Update the population
        for j in np.arange(i_time, min(i, len(time_points))):
            pop_out[j,:] = population_previous

        # Increment index
        i_time = i

    return pop_out

# Column 0 is change in x, column 1 is change in y, column 2 is change in z
limited_capacity_model = np.array(
    [  # dx, dy, dz
        [ 1,  0,  0],  # Damage
        [-1,  0, -1],  # Detection
        [ 0,  0,  1],  # Correct repair
        [ 0,  1,  1],  # Incorrect repair
    ],
    dtype=np.int64,
)

def plot_results(time_points, population, resolution, parameters, title='365 days of 1 pack per day', save=None):
    axes = plt.figure(layout="constrained",figsize=(12,8)).subplot_mosaic(
        """
        AA
        BC
        DE
        FF
        GG
        """
    )
    u,b,r1,r2,n,epsilon = parameters
    axes['A'].plot(time_points, population[:,0], 'r', label='x')
    axes['A'].set_xlim(0, 365)
    axes['A'].set_xlabel('Time (days)')
    axes['A'].set_ylabel('Unrepaired lesions')
    axes['B'].plot(time_points[:resolution], population[:resolution,0], 'r', label='x')
    axes['B'].set_xlim(0, 1)
    axes['B'].set_xlabel('First day')
    axes['B'].set_ylabel('Unrepaired lesions')
    axes['C'].plot(time_points[-resolution:], population[-resolution:,0], 'r', label='x')
    axes['C'].set_xlim(365-1, 365)
    axes['C'].set_xlabel('Last day')
    axes['D'].plot(time_points[:resolution], n-population[:resolution,2], 'g', label='n-z')
    axes['D'].plot(time_points[:resolution], n-r2*n/(r1*population[:resolution,0]+r2), 'y', label='n-zbar')
    axes['D'].set_xlim(0, 1)
    axes['D'].set_xlabel('First day')
    axes['D'].set_ylabel('Active repair')
    axes['E'].plot(time_points[-resolution:], n-population[-resolution:,2], 'g', label='n-z')
    axes['E'].plot(time_points[-resolution:], n-r2*n/(r1*population[-resolution:,0]+r2), 'y', label='n-zbar')
    axes['E'].set_xlim(365-1, 365)
    axes['E'].set_xlabel('Last day')
    axes['F'].plot(time_points, n-population[:,2], 'g', label='n-z')
    axes['F'].plot(time_points, n-r2*n/(r1*population[:,0]+r2), 'y', label='n-zbar')
    axes['F'].set_xlim(0, 365)
    axes['F'].set_xlabel('Time (days)')
    axes['F'].set_ylabel('Active repair')
    axes['G'].plot(time_points, population[:,1], 'b', label='y')
    axes['G'].set_xlim(0, 365)
    axes['G'].set_xlabel('Time (days)')
    axes['G'].set_ylabel('Repair errors')
    axes['A'].set_title(title)
    if save is not None:
        plt.savefig(save, dpi=300)
    plt.show()

def gamma(x, a, b):
    return np.power(b,a) / np.math.gamma(a) * np.power(x,a-1) * np.exp(-b*x)

def gamma_mixture(x, a1, b1, a2, b2, w1, w2):
    return (w1*gamma(x, a1, b1) + w2*gamma(x, a2, b2))/(w1+w2)

def x_result(x, params):
    n, r1, r2, alpha, beta = params 
    return gamma_mixture(x, alpha+1, beta, alpha, beta, n*r1*alpha, n*r2*beta)

def x_pmf(x, params):
    n, r1, r2, alpha, beta = params
    w1,w2 = n*r1*alpha, n*r2*beta 
    return (w1*sgamma.pdf(x, alpha+1, scale=1/beta)+w2*sgamma.pdf(x, alpha, scale=1/beta))/(w1+w2)

def x_cdf(x, params):
    n, r1, r2, alpha, beta = params
    w1,w2 = n*r1*alpha, n*r2*beta 
    return (w1*sgamma.cdf(x, alpha+1, scale=1/beta)+w2*sgamma.cdf(x, alpha, scale=1/beta))/(w1+w2)

def plot_pmf(population, resolution, parameters,loglog=False,max_x=0.99):
    u,b,r1,r2,n,epsilon = parameters
    alpha = u/(r1*n)
    beta = (n*r2-u*b)/(n*r2)/b
    fig = plt.figure(figsize=(4, 4))
    pop = np.sort(population[resolution:-resolution,0])
    h,b = np.histogram(pop,bins=range(0,pop[int(len(pop)*max_x)],1),density=True)
    plt.plot(x_pmf(b, (n, r1, r2, alpha, beta)),'k')
    plt.plot(b[:-1],h,'ro')
    if loglog:
        plt.plot(b,sgamma.pdf(b, alpha+1, scale=1/beta),'--',color='grey')
        plt.plot(b,sgamma.pdf(b, alpha, scale=1/beta),'--',color='grey')
        plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Unrepaired lesions')
    plt.ylabel('Distribution')
    plt.show()

def plot_cdf(population, resolution, parameters):
    u,b,r1,r2,n,epsilon = parameters
    alpha = u/(r1*n)
    beta = (n*r2-u*b)/(n*r2)/b
    fig = plt.figure(figsize=(4, 4))
    # compute cdf of x from data
    cdfx = np.sort(population[resolution:-resolution,0])
    #cdfy = np.arange(1,len(cdfx)+1)/float(len(cdfx))
    cdfy = np.arange(len(cdfx))/float(len(cdfx))
    plt.plot(cdfx,x_cdf(cdfx, (n, r1, r2, alpha, beta)),'k')
    plt.plot(cdfx,sgamma.cdf(cdfx, alpha+1, scale=1/beta),'--',color='grey')
    plt.plot(cdfx,sgamma.cdf(cdfx, alpha, scale=1/beta),'--',color='grey')
    plt.plot(cdfx,cdfy,'r')
    plt.xscale('log')
    plt.xlabel('Unrepaired lesions')
    plt.ylabel('Cumulative')
    plt.title('Mean = %.2f, Expected = %.2f' % (np.mean(population[resolution:-resolution,0]), (r2/r1+b)*u*b/(r2*n-u*b)))
    plt.show()
