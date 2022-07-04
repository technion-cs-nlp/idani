import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import logging
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertForTokenClassification
from data_utils import load_domain_data, load_task_data, data_dict
from pathlib import Path
from argparse import ArgumentParser
import sys
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np
import itertools
import copy
import matplotlib.pyplot as plt
import re
logging.basicConfig(level=logging.INFO)


def get_top_neurons_per_tag(weights, p):
    top_neurons_per_label = {}
    theta = weights
    num_of_neurons_per_label = torch.ceil(torch.tensor(theta.shape[1] * p / 100)).int().item()
    for l in range(theta.shape[0]):
        weights_for_label = theta[l]
        sorted_neurons_for_label = np.argsort(-np.abs(weights_for_label))
        top_neurons_per_label[l] = sorted_neurons_for_label[:num_of_neurons_per_label]
    return top_neurons_per_label


def get_top_neurons(weights):
    ordered_neurons = []
    for p in np.logspace(-1, np.log10(100), 100):
        tnpt = get_top_neurons_per_tag(weights, p)
        top_neurons = [neuron.item() for label, neurons in tnpt.items() for neuron in neurons]
        new_neurons = np.setdiff1d(top_neurons, ordered_neurons)
        ordered_neurons += new_neurons.tolist()
    return ordered_neurons


def linear(X_train, Y_train):
    cls = LogisticRegression(max_iter=1000)
    cls.fit(X_train, Y_train)
    # cls.predict(X_test)
    print('train acc:')
    print(cls.score(X_train, Y_train))
    weights = cls.coef_
    ranking = get_top_neurons(weights)
    return ranking


def get_means(X_train, Y_train):
    relevant_vals = {0, 1}
    embeddings_by_val = {val: [] for val in relevant_vals}
    for sample, domain in zip(X_train, Y_train):
        embeddings_by_val[domain].append(sample)
    avg_embeds_with_labels = {label: np.mean(embeds, axis=0) for label, embeds in embeddings_by_val.items()}
    return avg_embeds_with_labels


def get_diff_sum(arr):
    diff = np.zeros_like(arr[0])
    for couple in itertools.combinations(arr, 2):
        diff += np.abs(couple[0] - couple[1])
    return diff


def probeless(X_train, Y_train):
    avg_embeds_with_labels = get_means(X_train, Y_train)
    avg_embeds = list(avg_embeds_with_labels.values())
    diff = get_diff_sum(avg_embeds)
    ranking = np.argsort(diff)[::-1].tolist()
    return ranking


def lnscale(neurons_list, upper_bound:float, lower_bound=0):
    lnsp = np.logspace(np.log(upper_bound), np.log(1 / 1000 if lower_bound == 0 else lower_bound), 768, base=np.e)
    scores = np.array([lnsp[neurons_list.index(i)] if i in neurons_list else 0 for i in range(768)], dtype=np.float32)
    return scores


def pred_score(task, y_pred, y_true):
    if task == "aspect":
        return f1_score(y_pred=y_pred, y_true=y_true)
    elif task == "mnli":
        return f1_score(y_pred=y_pred, y_true=y_true, average="macro")
    else:
        return (y_pred == y_true).sum() / len(y_true)


def predict(task, cls, pooler, X, Y):
    if task == "aspect":
        preds = cls(torch.tensor(X).to(device)).argmax(dim=1).cpu().numpy()
    else:
        preds = cls(pooler(torch.tensor(X).to(device).unsqueeze(dim=1))).argmax(
            dim=1).cpu().numpy()
    acc = pred_score(task, preds, Y)
    return acc, preds


def translate(ranking, beta, neurons, means, X, add_noise=False):
    if len(neurons) == 30:
        print('here')
    alpha = lnscale(ranking, beta)[neurons]
    noise = np.zeros(len(neurons))
    if add_noise:
        noise = np.random.normal(0, 1, len(neurons))
    modification = alpha * (means[0][neurons] - means[1][neurons] + noise)
    relevant_features = copy.deepcopy(X)
    relevant_features[:, neurons] += modification
    # mod_dev = relevant_features
    return relevant_features


def split_test(X, Y, n):
    Y = np.array(Y, int)
    hist = np.bincount(Y)
    size_per_label = np.array(hist / X.shape[0] * n, int)
    # i = int(n / 2)
    X_per_label = [X[Y == i] for i in range(hist.shape[0])]
    Y_per_label = [Y[Y == i] for i in range(hist.shape[0])]
    X_sampled = np.concatenate([data[: i] for i, data in zip(size_per_label, X_per_label)])
    X_rest = np.concatenate([data[i:] for i, data in zip(size_per_label, X_per_label)])
    Y_sampled = np.concatenate([data[: i] for i, data in zip(size_per_label, Y_per_label)])
    Y_rest = np.concatenate([data[i:] for i, data in zip(size_per_label, Y_per_label)])
    return X_sampled, Y_sampled, X_rest, Y_rest


def changed_words(task, tar_domain, init_preds, current_preds, gold_labels):
    gold_labels = np.array(gold_labels)
    test_path = data_dict[task]['test_paths'][tar_domain]
    with open(test_path, 'rb') as f:
        raw_data = pickle.load(f)
    examples_per_word = {}
    if task == 'aspect':
        all_words = [w for s in raw_data[0] for w in s]
        for i, word in enumerate(all_words):
            if not word in examples_per_word.keys():
                examples_per_word[word] = []
            examples_per_word[word].append(i)
    else:
        for i, sent in enumerate(raw_data[0]):
            whole_text = sent if task == 'sentiment' else sent[0] + ' ' + sent[1]
            for word in re.findall(r'\w+', whole_text):
                lower_word = word.lower()
                if not lower_word in examples_per_word.keys():
                    examples_per_word[lower_word] = []
                if not i in examples_per_word[lower_word]:
                    examples_per_word[lower_word].append(i)
    score_diff_per_word = {}
    for word, examples in examples_per_word.items():
        if len(examples) < 5:
            continue
        relevant_init = init_preds[examples]
        relevant_current = current_preds[examples]
        if np.count_nonzero(relevant_init != relevant_current) < 2:
            continue
        relevant_gold = gold_labels[examples]
        init_score = pred_score(task, relevant_init, relevant_gold) * 100
        curr_score = pred_score(task, relevant_current, relevant_gold) * 100
        score_diff_per_word[word] = round(curr_score - init_score, 1)
    sorted_scores = sorted(score_diff_per_word.items(), key=lambda x: x[1], reverse=True)
    res_path = Path('results', task, src_domain, tar_domain)
    with open(Path(res_path, 'changed_words.txt'), 'w+') as f:
        for item in sorted_scores[:10]:
            f.write(f'{item}\n')
    return sorted_scores


ranking_func = {'probeless': probeless,
                'linear': linear}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-task', type=str)
    parser.add_argument('-src_domain', type=str)
    parser.add_argument('-tar_domain', type=str)
    parser.add_argument('-seed', type=int)
    parser.add_argument('-ranking_method', type=str)
    parser.add_argument('-data_size', type=int, default=-1)
    parser.add_argument('-beta', type=int, default=0)
    parser.add_argument('--estimate', default=False, action='store_true')
    args = parser.parse_args()
    task = args.task
    src_domain = args.src_domain
    tar_domain = args.tar_domain
    seed = args.seed
    ranking_method = args.ranking_method
    data_size = args.data_size
    beta = args.beta
    translation = True
    ensemble = False
    estimate = args.estimate
    trans_str = 'translation' if translation else 'ablation'
    beta_str = f'_{str(beta)}' if beta != 0 else f'_all_betas'
    torch.manual_seed(seed)
    source_domains = [src_domain]
    pkl_path = Path('idani-models', task, f'{source_domains[0]}')
    check_on_test = True
    X_train_domain, X_train_task, Y_train_domain, Y_train_task, X_test_domain, X_test_task, Y_test_domain, Y_test_task = \
        load_task_data(task, tar_domain, source_domains, check_on_test, data_size, seed)
    if estimate:
        sample_size = int(X_test_task.shape[0] * 0.05)
        X_test_task, Y_test_task, X_rest, Y_rest = split_test(X_test_task, Y_test_task, sample_size)

    set_str = 'test' if check_on_test else 'dev'
    res_file_dir = Path('results', task, f'{source_domains[0]}', tar_domain, set_str,
                        'all' if data_size == -1 else str(data_size), f'seed_{seed}')
    if not res_file_dir.exists():
        res_file_dir.mkdir(parents=True, exist_ok=True)
    tmp = False
    res_file_name = f'{trans_str}{beta_str}{"_ensemble_3" if ensemble else ""}{"_tmp" if tmp else ""}_{ranking_method}.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(Path(res_file_dir, res_file_name), 'w+') as f:
        sys.stdout = f

        # ranking = linear(X_train, Y_train)
        means = get_means(X_train_domain, Y_train_domain)
        ranking = ranking_func[ranking_method](X_train_domain, Y_train_domain)
        # ranking = probeless(X_train_domain, Y_train_domain)
        print(f'ranking: {ranking}')

        step = 1
        # cls_domain = LogisticRegression(max_iter=1000)
        # cls_domain.fit(X_train_domain, Y_train_domain)
        # print('train domain classification acc:')
        # print(cls_domain.score(X_train_domain, Y_train_domain))
        saved_models_path = Path(pkl_path, 'ft_models' if data_size == -1 else f'ft_models_{data_size}', f'seed_{seed}')
        models_idx = [int(d.name[len('checkpoint-'):]) for d in saved_models_path.glob('*')]
        max_idx = max(models_idx)
        model_path = Path(saved_models_path, f'checkpoint-{str(max_idx)}')
        model_type = BertForTokenClassification if task == "aspect" else BertForSequenceClassification
        model = model_type.from_pretrained(model_path.__str__()).to(
            device)
        model.eval()
        pooler = model.bert.pooler
        cls_task = model.classifier
        # cls_task = LogisticRegression(max_iter=1000)
        # cls_task.fit(X_train_task, Y_train_task)
        print(f'source domain task training classification {"f1" if task == "aspect" else "acc"}:')
        acc, preds = predict(task, cls_task, pooler, X_train_task, Y_train_task)
        # score = pred_score(task, preds, Y_train_task)
        print(acc)
        mod_dev = None
        domain_accs, task_accs, rest_task_accs = [], [], []
        sils = []
        init_preds = []
        best_res, best_beta, best_k = -1, -1, -1
        for curr_beta in range(1, 11):
            if beta != 0:
                curr_beta = beta
            modified_neurons = []
            for i in tqdm(range(0, len(ranking), step)):
                modified_neurons.extend(ranking[i - step:i] if i > 0 else [])
                print(f'modifying {i} neurons:')
                # print(modified_neurons)
                dev_copy = copy.deepcopy(X_test_task)

                if translation:
                    if ensemble:
                        mod_dev = []
                        for j in range(3):
                            np.random.seed(j)
                            mod_dev.append(translate(ranking, curr_beta, modified_neurons, means, dev_copy, add_noise=True))
                        np.random.seed(seed)
                    else:
                        mod_dev = translate(ranking, curr_beta, modified_neurons, means, dev_copy, add_noise=False)
                # print(f'dev domain classification acc after translation of {i} neurons:')
                # dev_domain_acc = cls_domain.score(dev_copy, Y_test_domain)
                # print(dev_domain_acc)
                # dev_task_acc = cls_task.score(mod_dev, Y_dev_task)
                print(f'dev task classification {"f1" if task == "aspect" else "acc"} after translation of {i} neurons:')
                if ensemble:
                    curr_preds = []
                    for j in range(3):
                        task_acc, preds = predict(task, cls_task, pooler, mod_dev[j], Y_test_task)
                        curr_preds.append(preds)
                    curr_preds = np.array(curr_preds)
                    preds = np.array([np.argmax(np.bincount(curr_preds[:, j])) for j in range(curr_preds.shape[1])])
                    task_acc = pred_score(task, preds, Y_test_task)
                else:
                    task_acc, preds = predict(task, cls_task, pooler, mod_dev, Y_test_task)
                if i == 0:
                    init_preds = preds
                else:
                    changed_preds = (init_preds != preds).nonzero()[0]
                # dev_task_acc = pred_score(task, preds, Y_test_task)
                print(task_acc)
                # domain_accs.append(dev_domain_acc)
                task_accs.append(task_acc)
                if task_acc > best_res or (task_acc == best_res and i < best_k):
                    best_beta = curr_beta
                    best_k = i
                    best_res = task_acc
                if estimate:
                    rest_copy = copy.deepcopy(X_rest)
                    mod_rest = translate(ranking, curr_beta, modified_neurons, means, rest_copy)
                    rest_acc, rest_preds = predict(task, cls_task, pooler, mod_rest, Y_rest)
                    print(f'rest task classification {"f1" if task == "aspect" else "acc"} after translation of {i} neurons:')
                    print(rest_acc)
                    rest_task_accs.append(rest_acc)
            if beta != 0:
                break
        if beta == 0:
            acc, preds = predict(task, cls_task, pooler, X_rest, Y_rest)
            print('left-out set init acc:')
            print(acc)
            mod_dev = translate(ranking, best_beta, ranking[:best_k], means, X_rest)
            print(f'chosen params: beta={best_beta}, k={best_k}')
            print(f'left-out set {"f1" if task == "aspect" else "acc"} after translation:')
            acc, preds = predict(task, cls_task, pooler, mod_dev, Y_rest)
            print(acc)

        else:
            init_acc = round(task_accs[0], 4)
            max_acc, argmax_acc = round(np.array(task_accs).max(), 4), np.array(task_accs).argmax()
            plt.plot(range(0, len(ranking), step), task_accs)
            title = f'beta={beta}_{ranking_method}'
            plt.title(f'{title}, initial acc: {init_acc}, max: ({argmax_acc}, {max_acc})')
            plt.legend(['task acc'])
            # plt.annotate(f'initial acc: {init_acc}')
            # plt.annotate(f'max: ({argmax_acc},{max_acc})')
            plt.savefig(Path(res_file_dir, title))
