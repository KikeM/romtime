from scipy.stats.distributions import uniform


def get_uniform_dist(min, max):

    loc = min
    scale = max - min

    return uniform(loc=loc, scale=scale)


def round_parameters(sample, num=2):

    rounded_dict = dict((k, round(v, num)) for (k, v) in sample.items())

    return rounded_dict


def round_parameter_list(param_list, num=2):

    rounded_list = [
        dict((k, round(v, num)) for (k, v) in d.items()) for d in param_list
    ]

    return rounded_list
