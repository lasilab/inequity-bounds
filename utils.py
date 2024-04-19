import numpy as np


def compute_plugin_bounds(g_probs, pi_probs, mu_probs, pg):
    '''
    Function to compute plugin bounds in partial identification setting
    '''
    
    pid_upper_plugin = (1 - g_probs) * pi_probs / pg
    pid_lower_plugin = pid_upper_plugin - (1 - g_probs) * (1 - mu_probs) / pg
    return pid_lower_plugin, pid_upper_plugin

def compute_bias_corrected_bounds(g_probs, pi_probs, mu_probs, pg, indicator_d0, indicator_d1, test_treatment, test_y):
    '''
    Function to compute bias-corrected bounds in partial identification setting
    '''

    pid_upper_bc = 1 / pg * ( pi_probs * indicator_d0 + indicator_d1 * (test_treatment - pi_probs) * g_probs / (1 - g_probs))
    pid_lower_bc = 1 / pg * ( indicator_d1 * (test_treatment - pi_probs) * g_probs / (1 - g_probs) + indicator_d0 * pi_probs + \
                              indicator_d0 * (test_y - 1))

    return pid_lower_bc, pid_upper_bc

def compute_gamma_plugin_bounds(gamma, gamma_prime, mu_probs, pi_probs, pg, g_probs):
    '''
    Function to compute plugin bounds in sensitivity analysis with gamma
    '''

    A = (gamma * mu_probs * pi_probs) / ((gamma - 1) * mu_probs + np.ones(len(mu_probs)))
    A_prime = (gamma_prime * mu_probs * pi_probs) / ((gamma_prime - 1) * mu_probs + np.ones(len(mu_probs)))

    upper_plugin_gamma = A * (1 / pg) * (1 - g_probs)
    lower_plugin_gamma = A_prime * (1 / pg) * (1 - g_probs)

    return lower_plugin_gamma, upper_plugin_gamma

def compute_gamma_bias_corrected_bounds(gamma, gamma_prime, mu_probs, pi_probs, pg, g_over_1_minus_g, indicator_d0, indicator_d1, test_treatment, test_y):
    '''
    Function to compute bias-corrected bounds in sensitivity analysis with gamma
    '''

    A = (gamma * mu_probs * pi_probs) / ((gamma - 1) * mu_probs + np.ones(len(mu_probs)))
    A_prime = (gamma_prime * mu_probs * pi_probs) / ((gamma_prime - 1) * mu_probs + np.ones(len(mu_probs)))

    upper_bc_gamma = 1 / pg * ( indicator_d0 * A \
        + indicator_d1 * gamma * mu_probs / ((gamma - 1) * mu_probs + np.ones(len(mu_probs))) * ( test_treatment - pi_probs) * g_over_1_minus_g \
        + indicator_d0 * gamma * pi_probs / ((gamma - 1) * mu_probs + np.ones(len(mu_probs)))**2 * ( test_y - mu_probs))

    lower_bc_gamma = 1 / pg * ( indicator_d0 * A_prime \
        + indicator_d1 * gamma_prime * mu_probs / ((gamma_prime - 1) * mu_probs + np.ones(len(mu_probs))) * ( test_treatment - pi_probs) * g_over_1_minus_g \
        + indicator_d0 * gamma_prime * pi_probs / ((gamma_prime - 1) * mu_probs + np.ones(len(mu_probs)))**2 * ( test_y - mu_probs))

    return lower_bc_gamma, upper_bc_gamma

def compute_lower_gamma_nonsmooth(pid_lower_plugin, pid_lower_bc, lower_plugin_gamma, lower_bc_gamma):

    '''
    Function to compute the gamma lower bound (applying max's)
    '''
    
    lower_bound_gamma = np.maximum(lower_plugin_gamma, pid_lower_plugin * np.ones(lower_plugin_gamma.shape)) # comparing with pid bound
    lower_bound_gamma = np.maximum(lower_bound_gamma, np.zeros(len(lower_bound_gamma))) # comparing with 0
    lower_bound_gamma[lower_bound_gamma == lower_plugin_gamma] = lower_bc_gamma[lower_bound_gamma == lower_plugin_gamma]
    lower_bound_gamma[lower_bound_gamma == pid_lower_plugin] = pid_lower_bc[lower_bound_gamma == pid_lower_plugin]
    return lower_bound_gamma

def compute_upper_gamma_nonsmooth(upper_plugin_gamma, upper_bc_gamma):

    '''
    Function to compute the gamma upper bound (applying min's)
    '''

    upper_bound_gamma = upper_bc_gamma.copy()
    upper_bound_gamma[upper_plugin_gamma > 1] = 1
    return upper_bound_gamma