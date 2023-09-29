import numpy as np
from poibin.poibin import PoiBin

def compute_TV_distance_by_prob_samples(p_vec, gamma_fn):
    rv = PoiBin(p_vec)
    K = len(gamma_fn) - 1
    mid = (K + 1) // 2
    prob_diff_vals = np.array([rv.pmf(l) - rv.pmf(K - l) for l in range(mid, K + 1)])
    half_gamma_fn = gamma_fn[mid:]
    obj_val = np.abs(0.5 * np.dot(prob_diff_vals, 1 - half_gamma_fn))
    return obj_val


def compute_estimated_TV_distance(gamma_fn, T=10000):
    est_obj_val = 0
    K = len(gamma_fn)
    for trial in range(T):
        # sample probabilities
        p_vec = np.random.random(size=K)
        # get objective value based on sampled probabilities
        obj_val = compute_TV_distance_by_prob_samples(p_vec, gamma_fn)
        est_obj_val += obj_val
    est_obj_val /= T
    return est_obj_val


def compute_composed_privacy_and_failure_prob_by_subsamples(eps, k, delta_0, delta_prime):
    final_eps = np.sqrt(2 * k * np.log(1 / delta_prime)) * eps + k * eps * (np.exp(eps) - 1)
    final_delta = k * delta_0 + delta_prime
    m = final_eps / eps
    print('k = {}, eps = {}, composed eps: {}, composed delta: {}, m: {}'.format(k, eps, final_eps, final_delta, m))
    return final_eps, final_delta, m

if __name__ == '__main__':
    compute_composed_privacy_and_failure_prob_by_subsamples(eps=0.1, k=5, delta_0=0.01, delta_prime=0.01)

