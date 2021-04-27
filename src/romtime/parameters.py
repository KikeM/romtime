from scipy.stats.distributions import uniform


def get_uniform_dist(min, max):

    loc = min
    scale = max - min

    return uniform(loc=loc, scale=scale)
