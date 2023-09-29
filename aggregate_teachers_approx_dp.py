import torch
from models.net_mnist import CNN
import os
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from optimize_gamma import optimize_gamma, optimize_gamma_gurobi
from tqdm import tqdm


def ask_teachers_to_label_data(list_private_teachers, dataloader, device='cpu'):
    # query private labels from teachers
    K = len(list_private_teachers)
    list_teacher_responses = []
    for teacher_idx in range(K):
        teacher = list_private_teachers[teacher_idx]
        teacher_response = []

        for images, _ in dataloader:
            images = images.to(device)
            test_output, last_layer = teacher(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            teacher_res = pred_y.cpu().detach().numpy()
            teacher_response.append(teacher_res)
        if len(teacher_response) == 1:
            teacher_response = teacher_response[0]
        else:
            teacher_response = np.concatenate(teacher_response)
        print(teacher_idx, teacher_response)
        list_teacher_responses.append(teacher_response)
    list_teacher_responses = np.vstack(list_teacher_responses)
    return list_teacher_responses


def prepare_dataloader_to_label(test_data, n_to_label=10, random=False):
    n_test_samples = len(test_data)
    assert (n_to_label <= n_test_samples)
    if not random:
        indices = range(n_to_label)
    else:
        indices = np.random.choice(n_test_samples, size=n_to_label, replace=False)

    test_image_to_label = torch.utils.data.Subset(test_data, indices)
    print('shape of test data to be labeled', len(test_image_to_label))
    dataloader = torch.utils.data.DataLoader(test_image_to_label,
                                            batch_size=32,
                                            shuffle=False,
                                            num_workers=1)
    return dataloader


def get_true_labels(dataloader):
    true_labels = []
    for _, lb in dataloader:
        lb = lb.cpu().numpy()
        true_labels.append(lb)
    if len(true_labels) == 1:
        true_labels = true_labels[0]
    else:
        true_labels = np.concatenate(true_labels)
    return true_labels

def get_disagree_samples(list_private_teachers, test_data, device='cpu'):
    dataloader = torch.utils.data.DataLoader(test_data,
                                            batch_size=32,
                                            shuffle=False,
                                            num_workers=1)
    all_teacher_preds = [[] for _ in range(len(list_private_teachers))]
    all_true_labels = []
    for batch_idx, (images, lbs) in enumerate(tqdm(dataloader)):
        all_true_labels.append(lbs.cpu().numpy())
        for t_idx, teacher in enumerate(list_private_teachers):
            images = images.to(device)
            test_output, last_layer = teacher(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            teacher_res = pred_y.cpu().detach().numpy()
            all_teacher_preds[t_idx].append(teacher_res)
    if len(all_true_labels) == 1:
        all_true_labels = all_true_labels[0]
    else:
        all_true_labels = np.concatenate(all_true_labels)
    for t_idx, t_preds in enumerate(all_teacher_preds):
        all_teacher_preds[t_idx] = np.concatenate(t_preds)
    all_teacher_preds = np.vstack(all_teacher_preds)
    print(all_teacher_preds.shape)
    all_diffs = np.zeros(all_teacher_preds.shape[1])
    for sample_idx in range(all_teacher_preds.shape[1]):
        teacher_resp = all_teacher_preds[:, sample_idx]
        unqa, counts = np.unique(teacher_resp, return_counts=True)
        if len(counts) <= 1:
            diff = 0
        else:
            diff = np.abs(counts[0] - counts[1])
        all_diffs[sample_idx] = diff
    non_zero_diff_idx = np.where(all_diffs > 0)[0]
    non_zero_diff = all_diffs[non_zero_diff_idx]
    # sorted_diff_idx = np.argsort(non_zero_diff)
    # non_zero_diff_idx = non_zero_diff_idx[sorted_diff_idx]
    # non_zero_diff = non_zero_diff[sorted_diff_idx]
    print(non_zero_diff_idx)
    print(len(non_zero_diff_idx))
    return all_teacher_preds, all_true_labels, non_zero_diff_idx, non_zero_diff



def aggregate_teacher_votes(teacher_responses, m, eps, delta_0, delta, aggregation_method='optimized'):
    eps_final = m * eps
    K, n_response = teacher_responses.shape
    aggregated_votes = []
    if aggregation_method == 'pate':
        log_1_delta = np.log(1/delta)
        lbd = log_1_delta / eps_final + 2
        diff = eps_final - log_1_delta / (lbd - 1)
        sigma = np.sqrt(lbd / diff)
        print('sigma: ', sigma)
        for i in range(n_response):
            teacher_answers = teacher_responses[:, i]
            class_5_counts = len(np.where(teacher_answers == 5)[0])
            class_8_counts = len(np.where(teacher_answers == 8)[0])
            # print('class 5 ct: {}, class 8 ct: {}'.format(class_5_counts, class_8_counts))
            noisy_class_5_counts = class_5_counts + np.random.normal(0, sigma)
            noisy_class_8_counts = class_8_counts + np.random.normal(0, sigma)
            # print('noisy class 5 ct: {}, noisy class 8 ct: {}'.format(noisy_class_5_counts, noisy_class_8_counts))
            if noisy_class_5_counts >= noisy_class_8_counts:
                vote = 5
            else:
                vote = 8
            aggregated_votes.append(vote)
    elif aggregation_method == 'optimized':
        opt_gamma = optimize_gamma_gurobi(K=K, eps=eps, m=m, delta=delta, delta_0=delta_0)
        for i in range(n_response):
            teacher_answers = teacher_responses[:, i]
            class_5_idx = np.where(teacher_answers == 5)[0]
            other_class_idx = np.setdiff1d(np.arange(K), class_5_idx)
            teacher_answers_reassign = np.zeros_like(teacher_answers)
            teacher_answers_reassign[other_class_idx] = 1
            L = np.sum(teacher_answers_reassign)
            prob = opt_gamma[L]
            if prob > 1:
                prob = 1
            elif prob < 0:
                prob = 0
            coin = np.random.binomial(1, prob)
            if coin == 1:  # output true answer with probability prob
                if L > K / 2:
                    vote = 8
                else:
                    vote = 5
            else:  # output a random bit
                vote = np.random.choice([5, 8])
            aggregated_votes.append(vote)
    elif aggregation_method == 'subsample':  # this is simple subsample
        for i in range(n_response):
            teacher_answers = teacher_responses[:, i]
            sel_teachers = np.random.choice(K, size=m, replace=False)
            vote = np.bincount(teacher_answers[sel_teachers]).argmax()
            aggregated_votes.append(vote)
    else:
        raise Exception('Unknown aggregation method {}!'.format(aggregation_method))
    return aggregated_votes


def main():
    # load a list of private teachers
    K = 11
    # eps = 0.1075
    eps = 0.2825
    m = 3
    # delta_0 = 0.001
    # save_folder = 'private_mnist_teachers'
    delta_0 = 1e-8
    save_folder = 'private_mnist_teachers_2'
    delta = m * delta_0
    # load K teachers and make them private

    # device = 'cuda:5'
    device = 'cpu'

    list_private_teachers = []
    for teacher_idx in range(K):
        save_path = os.path.join(save_folder, 'teacher_{}.model'.format(teacher_idx))
        private_teacher_model = CNN()
        private_teacher_model.load_state_dict(torch.load(save_path, map_location=device))
        list_private_teachers.append(private_teacher_model)

    # load test data
    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )
    # get test data from target classes
    idx_test = (test_data.targets == 5) | (test_data.targets == 8)
    test_data.data = test_data.data[idx_test]
    test_data.targets = test_data.targets[idx_test]
    print(test_data.data.size(), test_data.targets.size())  # torch.Size([1866, 28, 28]) torch.Size([1866])

    num_exp = 10
    all_acc = {}
    n_to_label = 100

    use_non_unanimous = False
    for exp_idx in range(num_exp):
        if not use_non_unanimous:
            dataloader = prepare_dataloader_to_label(test_data, n_to_label=n_to_label, random=True)
            teacher_responses = ask_teachers_to_label_data(list_private_teachers, dataloader)
            print('Exp : ', exp_idx)
            print('teacher responses: ', teacher_responses)

            true_labels = get_true_labels(dataloader)
            print('true labels: ', true_labels)
        else:  # use least confident samples
            all_teacher_preds, all_true_labels, non_zero_diff_idx, non_zero_diff = get_disagree_samples(
                list_private_teachers=list_private_teachers, test_data=test_data, device=device)
            sel_idx = np.random.choice(len(non_zero_diff_idx), size=n_to_label, replace=False)
            sel_non_zero_diff_idx = non_zero_diff_idx[sel_idx]
            print('sel non zero diff idx: ', sel_non_zero_diff_idx)
            print(non_zero_diff[sel_idx])
            true_labels = all_true_labels[sel_non_zero_diff_idx]
            teacher_responses = all_teacher_preds[:, sel_non_zero_diff_idx]
        aggregation_methods = ['pate', 'optimized', 'subsample']
        agg_results = {}
        for agg_method in aggregation_methods:
            aggregated_votes = aggregate_teacher_votes(teacher_responses=teacher_responses, m=m, eps=eps,
                                                       delta_0=delta_0, delta=delta, aggregation_method=agg_method)
            print(agg_method, aggregated_votes)
            agg_results[agg_method] = aggregated_votes
            acc = len(np.where(aggregated_votes == true_labels)[0]) / len(aggregated_votes)
            print('acc: {:.4f}'.format(acc))
            if agg_method not in all_acc:
                all_acc[agg_method] = [acc]
            else:
                all_acc[agg_method].append(acc)
    # extract votes
    for agg_method, list_acc in all_acc.items():
        vec_acc = np.array(list_acc)
        print('Method {}, acc: {:.4f} ({:.2f})'.format(agg_method, np.mean(vec_acc), np.std(vec_acc)))

    # get_disagree_samples(list_private_teachers, test_data)


if __name__ == '__main__':
    main()
