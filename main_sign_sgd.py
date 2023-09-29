import os.path
import numpy as np
import torch
from scipy.stats import norm
from utils import load_MNIST_from_target_class, split_data, load_MNIST_full, load_SVHN_full, load_CIFAR10_full
import argparse
from models.net_mnist import CNN
from models.net_svhn import SVHNCnnModel
from models.net_cifar10 import CIFAR10CNN
from collections import OrderedDict
from optimize_gamma import optimize_gamma, optimize_gamma_gurobi
import pickle


def extract_tensor_sign(vec, device='cpu'):
    # vec is a single tensor
    grad_sign = torch.sign(vec)
    n_zero_signs = torch.sum(grad_sign == 0).item()
    rand_sign = torch.randint(0, 2, (n_zero_signs, )) * 2 - 1
    rand_sign = rand_sign.to(device)
    grad_sign[grad_sign == 0] = rand_sign.float()
    return grad_sign

def one_bit_quantizer(grad, device='cpu'):
    res = []
    for g in grad:
        res.append(extract_tensor_sign(g, device).to(device))
    return res

def dp_sign_compressor(grad, eps, delta, Delta_2, device='cpu'):
    # NOTE this version does not converge
    sigma = Delta_2 / eps * np.sqrt(2 * np.log(1.25 / delta))
    # print('sigma: ', sigma)
    # gaussian_var = norm(loc=0, scale=1)
    grad_sign = []
    for g in grad:
        # normalized_g = g / sigma
        normalized_g = g.cpu().numpy() / sigma
        # print('normalized g: ', normalized_g[0])
        prob_vec = torch.Tensor(norm.cdf(normalized_g))
        # print('prob vec: ', prob_vec[0])
        res_sign = torch.bernoulli(prob_vec) * 2 - 1
        res_sign = res_sign.to(device)
        # print('prob vec: ', prob_vec)
        # res_sign = torch.Tensor([np.random.choice([1, -1], p=[prob, 1 - prob]) for prob in prob_vec.ravel()])\
        #     .reshape(g.size()).to(device)
        grad_sign.append(res_sign)
    return grad_sign


def beta_sign_sgd_compressor(grad, B, beta, device='cpu'):
    # clip grad
    grad_sign = []
    for g in grad:
        clipped_g = torch.max(torch.ones_like(g) * -B, torch.min(torch.ones_like(g) * B, g))
        probs = (torch.ones_like(g) * (B + beta) + clipped_g) / (2 * torch.ones_like(g) * (B + beta))
        res_sign = torch.bernoulli(probs) * 2 - 1
        res_sign = res_sign.to(device)
        grad_sign.append(res_sign)
    return grad_sign


def compute_grad_sum(list_grads):
    # note each grad is a tuple that aligns with the structure of the model
    n_samples = len(list_grads)
    n_blocks = len(list_grads[0])
    sum_grad = []
    for block_idx in range(n_blocks):
        grads_block = torch.stack([list_grads[i][block_idx] for i in range(n_samples)])
        sum_grad_block = torch.sum(grads_block, dim=0, keepdim=True)[0]
        sum_grad.append(sum_grad_block)
    return sum_grad

def clip_norm(grad, C):
    grad_norm = torch.norm(torch.concatenate([b.view(-1) for b in grad]), p=2).item()
    if grad_norm > C:  # clip gradient norm
        grad = [grad[block_idx] * C / grad_norm for block_idx in range(len(grad))]
    return grad

def worker_compute_grad(worker_dataloader, model, loss_fn, args, device='cpu', with_dp=False):
    # model should have been updated with current parameter w
    # args contains all parameters, including C, eps, delta
    all_grad = []
    for batch_idx, (images, labels) in enumerate(worker_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.C, norm_type=2)
        batch_grad = [p.grad for p in model.parameters()]
        all_grad.append(batch_grad)
    grad_worker = compute_grad_sum(all_grad)
    grad_worker = [grad_worker[block_idx] / len(worker_dataloader) for block_idx in range(len(grad_worker))]

    # now compress worker grad
    if with_dp:
        # dp compressor
        # print(grad_worker[0])
        # grad_worker = dp_sign_compressor(grad_worker, args.eps, args.delta, args.C, device=args.device)
        grad_worker = beta_sign_sgd_compressor(grad_worker, B=args.B, beta=args.beta, device=args.device)
    else:
        # compress non-dp
        # print(grad_worker[0])
        grad_worker = one_bit_quantizer(grad_worker, device=args.device)
    return grad_worker


def server_aggregate(list_worker_grads, args, device='cpu'):
    print('begin aggreation ...')
    block_len = len(list_worker_grads[0])
    mean_grads = []
    if args.aggregation == 'majority':  # non private aggregation
        for block_idx in range(block_len):
            list_block_grads = [list_worker_grads[worker_idx][block_idx] for worker_idx in range(len(list_worker_grads))]
            grad_mean = torch.mean(torch.stack(list_block_grads), 0).to(device)
            mean_grads.append(grad_mean)
        sign_grad = one_bit_quantizer(mean_grads, device=device)
    elif args.aggregation == 'subsampling':
        # private aggregation, need privacy budget m
        sel_worker_idx = np.random.choice(args.K, size=args.m, replace=False)
        print('sel worker idx: ', sel_worker_idx)
        for block_idx in range(block_len):
            list_block_grads = [list_worker_grads[worker_idx][block_idx] for worker_idx in sel_worker_idx]
            grad_mean = torch.mean(torch.stack(list_block_grads), 0).to(device)
            mean_grads.append(grad_mean)
        sign_grad = one_bit_quantizer(mean_grads, device=device)
    elif args.aggregation == 'laplacian':
        # similar to pate's aggregation, we add proper Laplacian noise to the aggregated output
        sign_grad = []
        lap_gamma = args.m * args.eps / 2
        for block_idx in range(block_len):
            list_block_grads_with_01 = []
            for worker_idx in range(len(list_worker_grads)):
                worker_grad = list_worker_grads[worker_idx][block_idx]
                worker_grad[worker_grad == -1] = 0
                list_block_grads_with_01.append(worker_grad)
            n_ones = torch.sum(torch.stack(list_block_grads_with_01), 0).to(device)
            n_zeros = len(list_block_grads_with_01) - n_ones
            lap_dist = torch.distributions.laplace.Laplace(torch.zeros_like(n_ones),
                                                           torch.ones_like(n_ones) * (1 / lap_gamma))
            ones_noise = lap_dist.sample().to(device)
            zeros_noise = lap_dist.sample().to(device)
            noisy_n_ones = n_ones + ones_noise
            noisy_n_zeros = n_zeros + zeros_noise
            sign_grad_block = torch.ones_like(n_ones)
            sign_grad_block[noisy_n_zeros > noisy_n_ones] = -1
            sign_grad.append(sign_grad_block)
    elif args.aggregation == 'optimized':
        sign_grad = []
        for block_idx in range(block_len):
            list_block_grads_with_01 = []
            for worker_idx in range(len(list_worker_grads)):
                worker_grad = list_worker_grads[worker_idx][block_idx]
                worker_grad[worker_grad == -1] = 0
                list_block_grads_with_01.append(worker_grad)
            n_ones = torch.sum(torch.stack(list_block_grads_with_01), 0).int().to(device)
            true_maj_sign = torch.ones_like(n_ones)
            true_maj_sign[n_ones <= args.K // 2] = -1
            probs = args.opt_gamma[n_ones]
            to_answer_true_majority = torch.bernoulli(probs).to(device)
            sign_grad_block = torch.zeros_like(n_ones)
            sign_grad_block[to_answer_true_majority == 1] = true_maj_sign[to_answer_true_majority == 1]
            rand_answer = (torch.randint(0, 2, n_ones.size()) * 2 - 1).int().to(device)
            sign_grad_block[to_answer_true_majority == 0] = rand_answer[to_answer_true_majority == 0]
            sign_grad.append(sign_grad_block)
    else:
        raise Exception('Unknown aggregation method {}!'.format(args.aggregation_method))
    return sign_grad


def test_accuracy(model, test_dataloader, device='cpu'):
    correct_samples = 0
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(test_dataloader):
        total_samples += len(labels)
        images = images.to(device)
        labels = labels.to(device)
        test_output = model(images)
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        correct_samples += (pred_y == labels).sum().item()
    acc = correct_samples / total_samples
    print('Acc: {:.4f}'.format(acc))
    return acc


## important utility functions
def _extract_model_parameters(model):
    params = [p.data for p in model.parameters()]  # list of torch tensors
    return params


def _set_model_parameter(model, new_parameter):
    new_dict = OrderedDict()
    for param_idx, param_name in enumerate(model.state_dict().keys()):
        new_dict[param_name] = new_parameter[param_idx]
    model.load_state_dict(new_dict)


def train_single_worker(worker_dataloader, loss_fn, args, test_dataloader=None, with_dp=False):
    # init model
    if args.dataname == 'mnist':
        model = CNN(n_classes=10).to(args.device)
    elif args.dataname == 'svhn':
        model = SVHNCnnModel().to(args.device)
    elif args.dataname == 'cifar10':
        model = CIFAR10CNN().to(args.device)
    else:
        raise Exception('Unknown dataset {}!'.format(args.dataname))
    # test before training
    if test_dataloader is not None:
        test_accuracy(model, test_dataloader, device=args.device)
    # extract parameter of the model
    w_current = _extract_model_parameters(model)
    for epoch_idx in range(args.epochs):
        worker_grad = worker_compute_grad(worker_dataloader, model, loss_fn, args, device=args.device, with_dp=with_dp)
        # update parameter
        w_new = []
        # lr = scheduler.get_lr()[0]
        for w_old, g in zip(w_current, worker_grad):
            w_new.append(w_old - args.eta * g)
        # scheduler.step()
        # update model
        _set_model_parameter(model, w_new)
        w_current = w_new
        # test model
        if args.dp:
            print('Epoch {}, lr {}, eps {}'.format(epoch_idx, args.eta, args.eps))
        else:
            print('Epoch {}, lr {}'.format(epoch_idx, args.eta))
        if test_dataloader is not None:
            test_accuracy(model, test_dataloader, device=args.device)
    return model


def train_multiple_workers(list_worker_dataloaders, loss_fn, args, test_dataloader=None, with_dp=False):
    # save test results per epoch
    all_test_acc = []
    # init a list of models
    if args.dataname == 'mnist':
        model = CNN(n_classes=10).to(args.device)
    elif args.dataname == 'svhn':
        model = SVHNCnnModel().to(args.device)
    elif args.dataname == 'cifar10':
        model = CIFAR10CNN().to(args.device)
    else:
        raise Exception('Unknown dataset {}!'.format(args.dataname))
    if args.init_folder is not None:
        if args.dataname == 'mnist':
            init_w_path = os.path.join(args.init_folder, 'CNN_init_param_exp_{}.model'.format(args.exp_no))
        elif args.dataname == 'svhn':
            init_w_path = os.path.join(args.init_folder, 'SVHNCNN_init_param_exp_{}.model'.format(args.exp_no))
        elif args.dataname == 'cifar10':
            init_w_path = os.path.join(args.init_folder, 'CIFAR10CNN_init_param_exp_{}.model'.format(args.exp_no))
        else:
            raise Exception('Unknown dataset {}!'.format(args.dataname))
        print('Loading initial model weight from {}'.format(init_w_path))
        model.load_state_dict(torch.load(init_w_path, map_location=args.device))
    K = len(list_worker_dataloaders)
    # test before training
    if test_dataloader is not None:
        acc = test_accuracy(model, test_dataloader, device=args.device)
        all_test_acc.append(acc)
    # extract parameter of the model
    w_current = _extract_model_parameters(model)
    for epoch_idx in range(args.epochs):
        list_worker_grads = []
        for worker_idx in range(K):
            print('Training worker: {}'.format(worker_idx))
            worker_dataloader = list_worker_dataloaders[worker_idx]
            worker_grad = worker_compute_grad(worker_dataloader, model, loss_fn, args, device=args.device, with_dp=with_dp)
            list_worker_grads.append(worker_grad)
        # aggregate worker grads
        aggregated_grad = server_aggregate(list_worker_grads, args, device=args.device)
        # update model parameter
        w_new = []
        for w_old, g in zip(w_current, aggregated_grad):
            w_new.append(w_old - args.eta * g)
        # update model
        _set_model_parameter(model, w_new)
        w_current = w_new
        if args.dp:
            print('Epoch {}, lr {}, eps {}'.format(epoch_idx, args.eta, args.eps))
        else:
            print('Epoch {}, lr {}'.format(epoch_idx, args.eta))
        if test_dataloader is not None:
            acc = test_accuracy(model, test_dataloader, device=args.device)
            all_test_acc.append(acc)
    return model, all_test_acc


def main():
    # get parametersg
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=11, help='# teachers / workers')
    # parser.add_argument('--C', type=float, default=4, help='Clipped norm size. Defines L2 sensitivity in DP-SignSGD.')
    parser.add_argument('--B', type=float, default=0.1, help='Parameter of beta-SGD')
    parser.add_argument('--eps', type=float, default=0.1, help='eps parameter in LDP')
    # parser.add_argument('--delta', type=float, default=1e-5, help='delta parameter in LDP')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eta', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dp', type=bool, default=True, help='whether to use DP SignSGD',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--aggregation', type=str, default='majority', help='Server aggregation method')
    parser.add_argument('--m', type=int, default=3, help='Privacy budget at the servers side')
    parser.add_argument('--save-folder', type=str, default=None, help='Save per epoch accuracy result')
    parser.add_argument('--exp-no', type=int, default=1, help='Experiment no.')
    parser.add_argument('--init-folder', type=str, default=None, help='Load initial model weight from init folder')
    parser.add_argument('--dataname', type=str, default='cifar10', help='Name of the dataset to train with SignSGD')
    args = parser.parse_args()

    args.beta = 2 * args.B / (np.exp(args.eps) - 1)
    print('Aggregation method: {}, K = {}, m = {}'.format(args.aggregation, args.K, args.m))
    print('B: {}, eps: {}, beta: {}'.format(args.B, args.eps, args.beta))

    # optimize the gamma function
    if args.aggregation == 'optimized':
        # find gamma_opt here
        args.opt_gamma = optimize_gamma(K=args.K, eps=args.eps, m=args.m, delta_0=0, delta=0)
        # args.opt_gamma = optimize_gamma_gurobi(K=args.K, eps=args.eps, m=args.m, delta_0=0, delta=0)
        args.opt_gamma = torch.Tensor(args.opt_gamma).to(args.device)

    # create result folder
    if args.save_folder is not None and not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)
    if args.save_folder is not None:
        print('Saving results to {}'.format(args.save_folder))

    loss_func = torch.nn.CrossEntropyLoss()

    # load data
    if args.dataname == 'mnist':
        # train_data, test_data = load_MNIST_from_target_class()
        train_data, test_data = load_MNIST_full()
    elif args.dataname == 'svhn':
        train_data, test_data = load_SVHN_full()
    elif args.dataname == 'cifar10':
        train_data, test_data = load_CIFAR10_full()
    else:
        raise Exception('Dataset {} not supported!'.format(args.dataname))
    # ready to split data
    worker_datasets = split_data(train_data, args.K)
    # create test dataloader
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=1)

    list_worker_dataloaders = []
    for worker_idx in range(args.K):
    # for worker_idx in range(3):  # for testing purpose only
        worker_dataloader = torch.utils.data.DataLoader(worker_datasets[worker_idx],
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=1)
        list_worker_dataloaders.append(worker_dataloader)
    # test_worker_data = worker_datasets[0]
    # worker_dataloader = torch.utils.data.DataLoader(test_worker_data,
    #                                                 batch_size=args.batch_size,
    #                                                 shuffle=True,
    #                                                 num_workers=1)
    #
    #
    # train_single_worker(worker_dataloader, loss_func, args, test_dataloader=test_dataloader, with_dp=args.dp)

    # test multiple workers
    model, all_test_acc = train_multiple_workers(list_worker_dataloaders=list_worker_dataloaders, loss_fn=loss_func, args=args,
                           test_dataloader=test_dataloader, with_dp=args.dp)
    if args.save_folder is not None:
        save_path = os.path.join(args.save_folder,
                                 'test_acc_agg_{}_epochs_{}_K_{}_m_{}_eps_{}_eta_{}_exp_{}'
                                 .format(args.aggregation, args.epochs, args.K, args.m, args.eps, args.eta, args.exp_no)
                                 .replace('.', '_') + '.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(all_test_acc, f)



def test_svhn():
    K = 11
    device = 'cpu'
    train_data, test_data = load_SVHN_full()
    # ready to split data
    worker_datasets = split_data(train_data, K)
    # create test dataloader
    model = SVHNCnnModel()
    worker_dataloader = torch.utils.data.DataLoader(worker_datasets[0],
                                                  batch_size=32,
                                                  shuffle=True,
                                                  num_workers=1)
    for batch_idx, (images, labels) in enumerate(worker_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        print(type(output))
        print(output.size())
        break


def test_gradient():
    model = CNN()
    K = 11
    loss_func = torch.nn.CrossEntropyLoss()
    train_data, test_data = load_MNIST_from_target_class()
    worker_datasets = split_data(train_data, K)
    worker_dataloader = torch.utils.data.DataLoader(worker_datasets[0],
                                                    batch_size=32,
                                                    shuffle=True,
                                                    num_workers=1)

    for batch_idx, (images, labels) in enumerate(worker_dataloader):
        output = model(images)
        loss = loss_func(output, labels)
        loss.backward()
        # extract gradients from loss
        grads = [p.grad for p in model.parameters()]
        grad_norm = torch.norm(torch.concatenate([b.view(-1) for b in grads]), p=2).item()
        print('grad norm: ', grad_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=2)
        normalized_grads = [p.grad for p in model.parameters()]
        grad_norm_2 = torch.norm(torch.concatenate([b.view(-1) for b in normalized_grads]), p=2).item()
        print('grad norm 2:', grad_norm_2)
        break

def test_optimizer_scheduler():
    from torch.optim.lr_scheduler import StepLR

    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    for i in range(100):
        print('step : {}, lr: {}'.format(i, scheduler.get_lr()[0]))
        scheduler.step()


def save_model_init_weights():
    # save_folder = '/content/drive/MyDrive/DP_maj_training/init_param'
    save_folder = 'cifar10_init_param'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    num_exp = 5
    for exp_idx in range(1, num_exp + 1):
        # model = CNN()
        model = CIFAR10CNN()
        save_path = os.path.join(save_folder, 'CIFAR10CNN_init_param_exp_{}.model'.format(exp_idx))
        torch.save(model.state_dict(), save_path)

def plot_results():
    import matplotlib.pyplot as plt

    # save_folder = 'signsgd_full'
    save_folder = 'signsgd_cifar10'
    K, m = 11, 3
    eps = 0.1
    # epochs = 100
    epochs = 300
    # eta = 0.001
    eta = 0.0005

    # num_exp = 1
    # all_exp_nos = [1]
    all_exp_nos = [1,2,3,4, 5]
    fontsize = 20
    fontsize_ticks = 20
    aggregation_methods = ['subsampling', 'laplacian', 'optimized']
    aggregation_methods_to_label = {
        'subsampling': 'Subsampling',
        'laplacian': 'Laplacian',
        'optimized': 'Optimized DaRRM$_{\gamma}$'
    }
    for aggregation in aggregation_methods:
        all_test_acc = []
        for exp_no in all_exp_nos:
            if aggregation == 'subsampling' and exp_no == 1:
                save_path = os.path.join(save_folder,
                                         'test_acc_agg_{}_epochs_{}_K_{}_m_{}_eps_{}_eta_{}_exp_{}'
                                         .format(aggregation, 400, K, m, eps, eta, exp_no)
                                         .replace('.', '_') + '.pkl')
            else:
                save_path = os.path.join(save_folder,
                                         'test_acc_agg_{}_epochs_{}_K_{}_m_{}_eps_{}_eta_{}_exp_{}'
                                         .format(aggregation, epochs, K, m, eps, eta, exp_no)
                                         .replace('.', '_') + '.pkl')
            with open(save_path, 'rb') as f:
                test_acc = pickle.load(f)
                test_acc = test_acc[:301]
            all_test_acc.append(test_acc)
        # avg test acc
        all_test_acc = np.vstack(all_test_acc)
        avg_test_acc = np.mean(all_test_acc, axis=0)
        std_test_acc = np.std(all_test_acc, axis=0)
        print(aggregation)
        print(avg_test_acc[-20:])
        # plt.errorbar(np.arange(epochs + 1), avg_test_acc, yerr=std_test_acc, label=aggregation_methods_to_label[aggregation])
        plt.plot(np.arange(epochs + 1), avg_test_acc, label=aggregation_methods_to_label[aggregation])
        plt.fill_between(np.arange(epochs + 1), avg_test_acc - std_test_acc, avg_test_acc + std_test_acc, alpha=0.1)
        # plt.yticks([0, 0.2, 0.4, 0.6, 0.8], [0, 0.2, 0.4, 0.6, 0.8], fontsize=fontsize_ticks)
        # plt.xticks([0, 20, 40, 60, 80, 100], [0, 20, 40, 60, 80, 100], fontsize=fontsize_ticks)
        plt.yticks([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], fontsize=fontsize_ticks)
        plt.xticks([0, 50, 100, 150, 200, 250, 300], [0, 50, 100, 150, 200, 250, 300], fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.xlabel('# Communication Rounds', fontsize=fontsize)
    plt.ylabel('Test Accuracy', fontsize=fontsize)
    plt.title('Private SignSGD on dataset CIFAR10'.format(K), fontsize=fontsize)
    plt.show()
    # plt.savefig('signsgd_K_{}_m_{}_cifar10.pdf'.format(K, m), bbox_inches='tight', pad_inches=0.1)
    # plt.close()


if __name__ == '__main__':
    # main()
    # save_model_init_weights()
    # test_gradient()
    # test_optimizer_scheduler()
    plot_results()
    # test_svhn()
