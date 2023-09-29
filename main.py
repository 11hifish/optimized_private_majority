from optimize_gamma import optimize_gamma, optimize_gamma_gurobi
from poly_gamma import baseline_gamma, RR_deta_independent_gamma
from utility import compute_estimated_TV_distance, compute_composed_privacy_and_failure_prob_by_subsamples
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


def plot_gamma(list_gamma_functions, gamma_function_names):
    assert (len(list_gamma_functions) == len(gamma_function_names))
    for gamma_fn, name in zip(list_gamma_functions, gamma_function_names):
        plt.plot(np.arange(len(gamma_fn)), gamma_fn, label=name)
    plt.legend()
    plt.grid()
    plt.title('Gamma Function')
    plt.xlabel('K')
    plt.show()

def plot_estimated_TV_distance(list_gamma_functions, gamma_function_names, no_exp=10):
    assert (len(list_gamma_functions) == len(gamma_function_names))
    all_results = np.zeros((len(list_gamma_functions), no_exp))
    for exp_idx in range(no_exp):
        for fn_idx in range(len(list_gamma_functions)):
            gamma_fn = list_gamma_functions[fn_idx]
            est_TV_distance = compute_estimated_TV_distance(gamma_fn)
            all_results[fn_idx, exp_idx] = est_TV_distance
    for fn_idx in range(len(list_gamma_functions)):
        mean_est_TV_dist = np.mean(all_results[fn_idx])
        std_est_TV_dist = np.std(all_results[fn_idx])
        print(gamma_function_names[fn_idx], mean_est_TV_dist, std_est_TV_dist)
        plt.scatter(fn_idx, mean_est_TV_dist)
        plt.errorbar(fn_idx, mean_est_TV_dist, yerr=std_est_TV_dist, label=gamma_function_names[fn_idx])
    plt.legend()
    plt.grid()
    plt.title('Error')
    plt.show()


def main_simple():
    K = 101
    eps = 0.1
    # m = 5
    # all_ms = [1, 3, 5, 7]
    all_ms = [10, 20, 30, 40]
    delta_0 = 0
    # delta_0 = 1e-5
    all_opt_gammas = []
    all_subsample_gammas = []
    all_indp_gamma = []
    for m in all_ms:
        delta = delta_0 * m

        opt_gamma_path = 'opt_gamma_K_{}_eps_{}_m_{}_pure_DP.pkl'.format(K, eps, m)
        if delta_0 == 0 and os.path.isfile(opt_gamma_path):
            with open(opt_gamma_path, 'rb') as f:
                opt_gamma = pickle.load(f)
            print('Loading pre-optimized opt gamma from {} ...'.format(opt_gamma_path))
        else:
            print('Optimizing gamma K = {}, eps = {}, m ={}, delta = {}, delta_0 = {}'.format(K, eps, m, delta, delta_0))
            opt_gamma = optimize_gamma_gurobi(K, eps, m, delta, delta_0)

        all_opt_gammas.append(opt_gamma)

        subsample_gamma = baseline_gamma(K, m)
        all_subsample_gammas.append(subsample_gamma)

        data_indp_gamma = RR_deta_independent_gamma(K=K, tau_eps=K * eps, m=m, eps=eps, lbda=K * delta_0, delta=delta)
        all_indp_gamma.append(data_indp_gamma)
    # plot everything together
    color_map = {
        'optimized': 'r',
        'subsampling': 'b',
        'data_indp': 'g'
    }
    legend_labels = {
        'optimized': 'Optimized DaRRM$_{\gamma}$ (Ours)',
        'subsampling': 'Subsampling (Baseline)',
        'data_indp': 'Randomized Response (Baseline)'
    }
    fontsize = 20
    get_legend = False
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    # fig, axes = plt.subplots(2, 2, figsize=(40, 40))
    for i in range(len(all_ms)):
        row_idx = i // 2
        col_idx = i % 2
        axes[row_idx, col_idx].plot(np.arange(K + 1), all_opt_gammas[i], c=color_map['optimized'], label=legend_labels['optimized'])
        axes[row_idx, col_idx].plot(np.arange(K + 1), all_subsample_gammas[i], c=color_map['subsampling'],
                                    label=legend_labels['subsampling'])
        axes[row_idx, col_idx].plot(np.arange(K + 1), all_indp_gamma[i], c=color_map['data_indp'],
                                    label=legend_labels['data_indp'])
        axes[row_idx, col_idx].set_title('m = {}'.format(all_ms[i]), fontsize=fontsize)
        axes[row_idx, col_idx].set_xticks([0, K / 2, K], [0, K / 2, K], fontsize=fontsize)
        axes[row_idx, col_idx].set_yticks([0,0.5, 1], [0, 0.5, 1], fontsize=fontsize)
        axes[row_idx, col_idx].grid()
        if get_legend and i == 3:
            legend_fontsize = 15
            legend = axes[row_idx, col_idx].legend(fontsize=legend_fontsize, ncol=3, loc='lower right')
    for ax in fig.get_axes():
        ax.label_outer()
    fig.text(0.5, 0.01, 'Support $\mathcal{L}$', ha='center', fontsize=fontsize)
    fig.text(0.04, 0.5, '$\gamma$ values', va='center', rotation='vertical', fontsize=fontsize)
    fig.suptitle('$\gamma$ functions', fontsize=fontsize)

    dp_str = '_pure_DP' if delta_0 == 0 else '_approx_DP'
    plt.savefig('gamma_fn_K_{}{}.pdf'.format(K, dp_str), bbox_inches='tight', pad_inches=0.1)
    plt.close()
    # plt.show()

    def export_legend(legend, filename="legend.png", expand=[-5, -5, 5, 5]):
        for legobj in legend.legendHandles:
            legobj.set_linewidth(2)
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    if get_legend:
        export_legend(legend, filename='legend.png')


    # no_exp = 10
    # fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    # ## plot estimated TV distance
    # for i in range(len(all_ms)):
    #     row_idx = i // 2
    #     col_idx = i % 2
    #     axes[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=fontsize)
    #     axes[row_idx, col_idx].tick_params(axis='both', which='minor', labelsize=fontsize)
    #     # compute estimated TV distance
    #     list_gamma_functions = [all_opt_gammas[i], all_subsample_gammas[i], all_indp_gamma[i]]
    #     color_map_2 = [color_map['optimized'], color_map['subsampling'], color_map['data_indp']]
    #     all_results = np.zeros((len(list_gamma_functions), no_exp))
    #     for exp_idx in range(no_exp):
    #         for fn_idx in range(len(list_gamma_functions)):
    #             gamma_fn = list_gamma_functions[fn_idx]
    #             est_TV_distance = compute_estimated_TV_distance(gamma_fn)
    #             all_results[fn_idx, exp_idx] = est_TV_distance
    #     for fn_idx in range(len(list_gamma_functions)):
    #         mean_est_TV_dist = np.mean(all_results[fn_idx])
    #         std_est_TV_dist = np.std(all_results[fn_idx])
    #         # plot avg error of a specific funtion
    #         axes[row_idx, col_idx].scatter(fn_idx, mean_est_TV_dist, c=color_map_2[fn_idx])
    #         axes[row_idx, col_idx].errorbar(fn_idx, mean_est_TV_dist, yerr=std_est_TV_dist, c=color_map_2[fn_idx])
    #         axes[row_idx, col_idx].set_xticks([])
    #     axes[row_idx, col_idx].set_title('m = {}'.format(all_ms[i]), fontsize=fontsize)
    #     axes[row_idx, col_idx].grid()
    # # fig.text(0.5, 0.01, 'Error', ha='center', fontsize=fontsize)
    # # fig.text(0.04, 0.5, '$\gamma$ functions', va='center', rotation='vertical', fontsize=fontsize)
    # fig.text(0.5, 0.01, '$\gamma$ functions', ha='center', fontsize=fontsize)
    # fig.text(0.01, 0.5, 'TV Distance', va='center', rotation='vertical', fontsize=fontsize)
    # fig.suptitle('Error', fontsize=fontsize)
    # # plt.show()
    # dp_str = '_pure_DP' if delta_0 == 0 else '_approx_DP'
    # plt.savefig('error_K_{}{}.pdf'.format(K, dp_str), bbox_inches='tight', pad_inches=0.1)
    # plt.close()

    ## old version of plotting scripts
    # plot_gamma([opt_gamma, subsample_gamma], ['optimized', 'simple subsampling'])

    # plot estimated TV distance
    # plot_estimated_TV_distance([opt_gamma, subsample_gamma], ['optimized', 'simple subsampling'])


def main_advanced():
    def compute_m(k, delta_prime, eps):
        return np.sqrt(2 * k * np.log(1 / delta_prime)) + k * (np.exp(eps) - 1)

    K = 35
    eps = 0.1
    all_Ms = [10, 13, 15, 20]
    delta_0 = 1e-5
    delta_prime = 0.1
    all_opt_gammas = []
    all_subsample_gammas = []
    all_indp_gamma = []
    for M in all_Ms:
        delta = M * delta_0 + delta_prime

        if M == 10:
            opt_gamma_path = 'opt_gamma_K_35_eps_0.1_m_7.83784960517159_pure_DP.pkl'
        elif M == 13:
            opt_gamma_path = 'opt_gamma_K_35_eps_0.1_m_9.104612478173365_pure_DP.pkl'
        elif M == 15:
            opt_gamma_path = 'opt_gamma_K_35_eps_0.1_m_9.888854452480267_pure_DP.pkl'
        elif M == 20:
            opt_gamma_path = 'opt_gamma_K_35_eps_0.1_m_11.700470185889117_pure_DP.pkl'
        else:
            raise Exception('Unsupported subsampling size M = {}!'.format(M))

        with open(opt_gamma_path, 'rb') as f:
            opt_gamma = pickle.load(f)
        all_opt_gammas.append(opt_gamma)

        subsample_gamma = baseline_gamma(K, M)
        all_subsample_gammas.append(subsample_gamma)

        # compute parameters for data-independent gamma
        tau_eps = np.sqrt(2 * K * np.log(1 / delta_prime)) * eps + K * eps * (np.exp(eps) - 1)
        lbda = K * delta_0 + delta_prime
        m = compute_m(M, delta_prime, eps)
        data_indp_gamma = RR_deta_independent_gamma(K=K, tau_eps=tau_eps, m=m, eps=eps, lbda=lbda, delta=delta)
        all_indp_gamma.append(data_indp_gamma)
    # plot everything together
    color_map = {
        'optimized': 'r',
        'subsampling': 'b',
        'data_indp': 'g'
    }
    legend_labels = {
        'optimized': 'Optimized DaRRM$_{\gamma}$ (Ours)',
        'subsampling': 'Subsampling (Baseline)',
        'data_indp': 'Randomized Response (Baseline)'
    }
    fontsize = 20
    get_legend = False
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    # fig, axes = plt.subplots(2, 2, figsize=(40, 40))
    for i in range(len(all_Ms)):
        row_idx = i // 2
        col_idx = i % 2
        axes[row_idx, col_idx].plot(np.arange(K + 1), all_opt_gammas[i], c=color_map['optimized'],
                                    label=legend_labels['optimized'])
        axes[row_idx, col_idx].plot(np.arange(K + 1), all_subsample_gammas[i], c=color_map['subsampling'],
                                    label=legend_labels['subsampling'])
        axes[row_idx, col_idx].plot(np.arange(K + 1), all_indp_gamma[i], c=color_map['data_indp'],
                                    label=legend_labels['data_indp'])
        axes[row_idx, col_idx].set_title('M = {}'.format(all_Ms[i]), fontsize=fontsize)
        axes[row_idx, col_idx].set_xticks([0, K / 2, K], [0, K / 2, K], fontsize=fontsize)
        axes[row_idx, col_idx].set_yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=fontsize)
        axes[row_idx, col_idx].grid()
        if get_legend and i == 3:
            legend_fontsize = 15
            legend = axes[row_idx, col_idx].legend(fontsize=legend_fontsize, ncol=3, loc='lower right')
    for ax in fig.get_axes():
        ax.label_outer()
    fig.text(0.5, 0.01, 'Support $\mathcal{L}$', ha='center', fontsize=fontsize)
    fig.text(0.04, 0.5, '$\gamma$ values', va='center', rotation='vertical', fontsize=fontsize)
    fig.suptitle('$\gamma$ functions', fontsize=fontsize)

    dp_str = '_pure_DP' if delta_0 == 0 else '_approx_DP'
    # plt.show()
    plt.savefig('gamma_fn_K_{}{}.pdf'.format(K, dp_str), bbox_inches='tight', pad_inches=0.1)
    plt.close()

    no_exp = 10
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ## plot estimated TV distance
    for i in range(len(all_Ms)):
        row_idx = i // 2
        col_idx = i % 2
        axes[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=fontsize)
        axes[row_idx, col_idx].tick_params(axis='both', which='minor', labelsize=fontsize)
        # compute estimated TV distance
        list_gamma_functions = [all_opt_gammas[i], all_subsample_gammas[i], all_indp_gamma[i]]
        color_map_2 = [color_map['optimized'], color_map['subsampling'], color_map['data_indp']]
        all_results = np.zeros((len(list_gamma_functions), no_exp))
        for exp_idx in range(no_exp):
            for fn_idx in range(len(list_gamma_functions)):
                gamma_fn = list_gamma_functions[fn_idx]
                est_TV_distance = compute_estimated_TV_distance(gamma_fn)
                all_results[fn_idx, exp_idx] = est_TV_distance
        for fn_idx in range(len(list_gamma_functions)):
            mean_est_TV_dist = np.mean(all_results[fn_idx])
            std_est_TV_dist = np.std(all_results[fn_idx])
            # plot avg error of a specific funtion
            axes[row_idx, col_idx].scatter(fn_idx, mean_est_TV_dist, c=color_map_2[fn_idx])
            axes[row_idx, col_idx].errorbar(fn_idx, mean_est_TV_dist, yerr=std_est_TV_dist, c=color_map_2[fn_idx])
            axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_title('M = {}'.format(all_Ms[i]), fontsize=fontsize)
        axes[row_idx, col_idx].grid()
    # fig.text(0.5, 0.01, 'Error', ha='center', fontsize=fontsize)
    # fig.text(0.04, 0.5, '$\gamma$ functions', va='center', rotation='vertical', fontsize=fontsize)
    fig.text(0.5, 0.01, '$\gamma$ functions', ha='center', fontsize=fontsize)
    fig.text(0.01, 0.5, 'TV Distance', va='center', rotation='vertical', fontsize=fontsize)
    fig.suptitle('Error', fontsize=fontsize)
    # plt.show()
    dp_str = '_pure_DP' if delta_0 == 0 else '_approx_DP'
    plt.savefig('error_K_{}{}.pdf'.format(K, dp_str), bbox_inches='tight', pad_inches=0.1)
    plt.close()




def main_advanced_depracated():
    opt_gamma_folder = 'opt_gamma_fn'
    if not os.path.isdir(opt_gamma_folder):
        os.mkdir(opt_gamma_folder)
    K = 51
    eps = 0.1
    delta_prime = 0.01
    delta_0 = 0.01
    k = 15
    final_eps, delta, m = \
        compute_composed_privacy_and_failure_prob_by_subsamples(eps=eps, k=k, delta_0=delta_0, delta_prime=delta_prime)
    print('K : {}, delta_0: {}, k: {}'.format(K, delta_0, k))
    print('final eps: {}, delta: {}, m: {}'.format(final_eps, delta, m))
    opt_gamma_path = os.path.join(opt_gamma_folder,
                                  'opt_gamma_K_{}_delta_0_{}_eps_{}_k_{}.pkl'.format(K, delta_0, eps, k)
                                  .replace('.', '_'))
    if not os.path.isfile(opt_gamma_path):
        opt_gamma = optimize_gamma(K, eps, m, delta, delta_0)

        with open(opt_gamma_path, 'wb') as f:
            pickle.dump(opt_gamma, f)
    else:
        with open(opt_gamma_path, 'rb') as f:
            opt_gamma = pickle.load(f)




    # subsample_gamma = baseline_gamma(K, k)
    #
    # plot_gamma([opt_gamma, subsample_gamma], ['optimized', 'advanced subsampling'])
    #
    # # plot estimated TV distance
    # plot_estimated_TV_distance([opt_gamma, subsample_gamma], ['optimized', 'advanced subsampling'])


def compute_and_save_gamma():
    def compute_m(k, delta_prime, eps):
        return np.sqrt(2 * k * np.log(1 / delta_prime)) + k * (np.exp(eps) - 1)

    K = 35
    eps = 0.1
    all_Ms = [10, 13, 15, 20]
    delta_0 = 1e-5
    delta_prime = 0.1
    all_ms = [compute_m(k=M, delta_prime=delta_prime, eps=eps) for M in all_Ms]
    print('m: ', all_ms)
    for M in all_Ms:
        m = compute_m(k=M, delta_prime=delta_prime, eps=eps)
        delta = M * delta_0 + delta_prime
        print('m = {}, m eps: {}, delta: {}'.format(m, m * eps, delta))
        # optimized_gamma = optimize_gamma_gurobi(K, eps, m, delta, delta_0)
        # with open('opt_gamma_K_{}_eps_{}_m_{}_pure_DP.pkl'.format(K, eps, m), 'wb') as f:
        #     pickle.dump(optimized_gamma, f)


if __name__ == '__main__':
    # main_simple()
    main_advanced()
    # compute_and_save_gamma()
