from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import pickle

finetuned = {
    'sentiment': {
        'all': {
            'airline': [88.0, 88.0, 87.6, 88.0, 88.3],
            'books': [88.0, 87.0, 87.0, 87.6, 90.3],
            'dvd': [88.3, 91.3, 88.3, 89.0, 89.0],
            'electronics': [92.3, 92.0, 93.0, 92.0, 92.6],
            'kitchen': [92.6, 93.0, 91.0, 92.3, 93.3]
        },
        '400': {
            'airline': [85.0, 88.0, 87.0, 86.3, 87.6],
            'books': [86.6, 84.6, 86.6, 87.0, 84.6],
            'dvd': [87.3, 87.3, 89.3, 89.0, 89.0],
            'electronics': [91.3, 91.6, 92.6, 90.6, 90.3],
            'kitchen': [89.6, 89.6, 90.6, 91.3, 89.6]
        },
        '100': {
            'airline': [84.0, 83.6, 82.3, 68.3, 78.3],
            'books': [80.6, 85.0, 84.6, 82.0, 86.3],
            'dvd': [71.6, 87.6, 72.6, 67.0, 85.0],
            'electronics': [85.6, 84.0, 87.6, 83.3, 83.6],
            'kitchen': [85.6, 85.3, 86.3, 79.6, 76.3]
        }
    },
    'aspect': {
        'all': {
            'device': [65.7, 66.2, 68.4, 68.6, 66.6],
            'laptops': [85.8, 84.9, 84.5, 86.4, 85.8],
            'rest': [84.2, 83.3, 83.7, 82.9, 83.1],
            'service': [82.0, 81.0, 81.2, 81.1, 80.7]
        }
    },
    'mnli': {
        'all': {
            'fiction': [65.1, 69.7, 64.9, 68.7, 69.4],
            'government': [73.5, 72.7, 73.7, 74.0, 75.3],
            'slate': [61.7, 64.2, 58.9, 64.2, 64.1],
            'telephone': [66.4, 68.9, 68.7, 68.8, 68.7],
            'travel': [69.2, 71.0, 69.7, 69.5, 70.0]
        }
    },
    'mnli_': {
        'all': {
            'fiction': [79.3, 80.0, 80.1, 81.2, 81.2],
            'government': [83.8, 83.7, 83.4, 83.9, 83.6],
            'slate': [76.3, 77.3, 75.8, 77.2, 76.6],
            'telephone': [80.2, 79.1, 80.6, 80.8, 80.2],
            'travel': [81.4, 81.7, 81.5, 81.9, 82.6]
        }
    }
}


def get_max_diff(accs: np.array):
    # return accs.argmax(), round(100 * accs.max(), 1)
    return round(100 * accs.max(), 1)


def process_file(p: Path, all_betas=False):
    betas = list(range(1, 11))
    accs = [] if not all_betas else {b: [] for b in betas}
    acc_line = False
    curr_beta = 0
    set_name = 'rest' if all_betas else 'dev'
    with open(p, 'r') as f:
        for line in f.readlines():
            if acc_line:
                if all_betas:
                    accs[curr_beta].append(round(float(line) * 100, 1))
                else:
                    accs.append(round(float(line) * 100, 1))
                acc_line = False
            if line.startswith(f'{set_name} task'):
                acc_line = True
            if line.startswith('modifying 0'):
                curr_beta += 1
    return np.array(accs) if not all_betas else {b: np.array(accs[b]) for b in betas}


def deltas(p: Path, oracle, all_betas=False, beta=-1):
    accs = process_file(p, all_betas)
    rest_acc = None
    if all_betas:
        accs = accs[beta]
    init = accs[0]
    if all_betas:
        with open(p, 'r') as f:
            lines = f.read().splitlines()
            rest_acc = round(float(lines[-1]) * 100, 1)
            rest_acc = round(rest_acc - init, 1)
    accs[1:] = [acc - init for acc in accs[1:]]
    upper_bound = oracle - init
    accs[0] = 0
    return init, accs, upper_bound, rest_acc


def best_beta(task: str, src: str, tar: str, data_size: int, custom_beta: int = 1, custom_k: int = 0):
    res_dir = Path('results', task, src, tar, 'test', str(data_size) if data_size != -1 else '')
    max_accs = []
    custom_acc = None
    init_acc = 0
    for beta in range(1, 21):
        file_path = Path(res_dir, f'translation_{str(beta)}.txt')
        accs = process_file(file_path)
        init_acc = round(100 * accs[0], 1)
        max_accs.append(get_max_diff(accs))
        if beta == custom_beta:
            custom_acc = 100 * accs[custom_k]
    # only_accs = np.array([acc for _, acc in max_accs])
    best_b = np.array(max_accs).argmax()
    max_acc = max_accs[best_b]
    return init_acc, max_acc, f'\u0394={round(max_acc - init_acc, 1)}', f'\u03B2={best_b + 1}'


def run_task(task):
    beta, k = 8, 50
    res = {}
    root_path = Path('results', task)
    for s in root_path.glob('*'):
        if not s.is_dir():
            continue
        src = s.name
        res[src] = {}
        for t in s.glob('*'):
            if not t.is_dir():
                continue
            tar = t.name
            res[src][tar] = []
            for data_size in [400]:
                curr_res = best_beta(task, src, tar, data_size, beta, k)
                res[src][tar].append(curr_res)
                print(src, tar, data_size * 2)
                print(curr_res[0], curr_res[1], curr_res[2], curr_res[3], sep=', ')
    return res


def probeless_vs_linear(all_betas, ensemble):
    const_beta, const_k = 8, 50
    ensemble_str = 'ensemble_' if ensemble else ''
    # tasks = ['sentiment', 'aspect', 'mnli']
    tasks = ['sentiment', 'aspect']
    res = {}
    for task in tasks:
        root_path = Path('results', task)
        for s in root_path.glob('*'):
            if not s.is_dir():
                continue
            src = s.name
            if not (src == 'airline' or src == 'rest'):
                continue
            res[src] = {}
            for t in tqdm(s.glob('*')):
                if not t.is_dir():
                    continue
                tar = t.name
                if not (tar == 'dvd' or tar == 'service'):
                    continue
                res[src][tar] = {}
                for data_size in ['all']:
                    curr_dir = Path(root_path, src, tar, 'test', data_size)
                    comp_dir = Path(curr_dir, 'avg', 'sampled' if all_betas else '')
                    if not comp_dir.exists():
                        comp_dir.mkdir()
                    probeless_oracle, linear_oracle = dict.fromkeys(range(1,6), 0), dict.fromkeys(range(1,6), 0)
                    finetuned_accs = finetuned[task][data_size]
                    # for beta in range(1, 11):
                    for beta in [8]:
                        exists_all = True
                        probeless_accs, linear_accs, init_accs, upper_bounds, probeless_selected, linear_selected =\
                            [], [], [], [], [], []
                        for seed in range(1, 6):
                            seed_dir = Path(curr_dir, f'seed_{seed}')
                            probeless_name = f'translation_all_betas_{ensemble_str}probeless.txt' if all_betas else \
                                f'translation_{str(beta)}_probeless.txt'
                            probeless_path = Path(seed_dir, probeless_name)
                            if not probeless_path.exists():
                                exists_all = False
                                break
                            curr_init, curr_probeless_acc, curr_upper_bound, curr_probeless_rest = \
                                deltas(probeless_path, finetuned_accs[tar][seed - 1], all_betas, beta)
                            probeless_accs.append(curr_probeless_acc)
                            probeless_selected.append(curr_probeless_rest)
                            init_accs.append(curr_init)
                            probeless_oracle[seed] = max(probeless_oracle[seed], max(curr_probeless_acc))
                            upper_bounds.append(curr_upper_bound)
                            linear_name = 'translation_all_betas_linear.txt' if all_betas else \
                                f'translation_{str(beta)}_linear.txt'
                            linear_path = Path(seed_dir, linear_name)
                            if not linear_path.exists():
                                exists_all = False
                                break
                            linear_res = deltas(linear_path, finetuned_accs[tar][seed - 1], all_betas, beta)
                            curr_linear_acc = linear_res[1]
                            linear_accs.append(curr_linear_acc)
                            curr_linear_rest = linear_res[3]
                            linear_selected.append(curr_linear_rest)
                            linear_oracle[seed] = max(linear_oracle[seed], max(curr_linear_acc))
                        if not exists_all:
                            break
                        probeless_accs, linear_accs, init_accs = \
                            np.array(probeless_accs), np.array(linear_accs), np.array(init_accs)
                        probeless_mean = np.mean(probeless_accs, axis=0)
                        probeless_std = np.std(probeless_accs, axis=0)
                        plt.plot(range(768), probeless_mean, label='Probeless')
                        # plt.fill_between(range(768), probeless_mean - probeless_std, probeless_mean + probeless_std, alpha=0.2)
                        linear_mean = np.mean(linear_accs, axis=0)
                        linear_std = np.std(linear_accs, axis=0)
                        plt.plot(range(768), linear_mean, label='Linear')
                        # plt.fill_between(range(768), linear_mean - linear_std, linear_mean + linear_std, alpha=0.2)
                        plt.axhline(0., color='black', linestyle='dashed', label='Init')
                        # plt.axhline(np.mean(upper_bounds), color='gray', linestyle='dashed', label='UB')
                        plt.xlabel('k')
                        plt.ylabel('\u0394')
                        rightarrow = u"\u2192"
                        plt.title(f'{src} {rightarrow} {tar}, \u03B2={str(beta)}')
                        plt.legend()
                        # plt.savefig(Path(comp_dir, f'beta={str(beta)}'))
                        plt.close()
                        if beta == const_beta:
                            res[src][tar]['probeless'] = (round(probeless_mean[const_k], 1), round(probeless_std[const_k], 1))
                            res[src][tar]['linear'] = (round(linear_mean[const_k], 1), round(linear_std[const_k], 1))
                            res[src][tar]['probeless_all'] = probeless_mean
                            res[src][tar]['linear_all'] = linear_mean
                    res[src][tar]['init'] = (round(np.mean(init_accs, axis=0), 1), round(np.std(init_accs, axis=0), 1))
                    res[src][tar]['UB'] = (round(np.mean(finetuned_accs[tar], axis=0), 1), round(np.std(finetuned_accs[tar], axis=0), 1))
                    res[src][tar]['probeless_oracle'] = round(np.mean(list(probeless_oracle.values())), 1), round(np.std(list(probeless_oracle.values())), 1)
                    res[src][tar]['probeless_selected'] = round(np.mean(probeless_selected), 1), round(np.std(probeless_selected), 1)
                    res[src][tar]['linear_oracle'] = round(np.mean(list(linear_oracle.values())), 1), round(
                        np.std(list(linear_oracle.values())), 1)
                    res[src][tar]['linear_selected'] = round(np.mean(linear_selected), 1), round(
                        np.std(linear_selected), 1)
                        # if beta == const_beta:
                        #     res[src][tar] = (probeless_accs[k], linear_accs[k])
    rightarrow = u"\u2192"
    for src in res.keys():
        for tar in res[src].keys():
            # linestyle = 'solid' if src != 'airline' else 'dashed'
            color = 'C0' if src != 'airline' else 'C1'
            plt.plot(range(768), res[src][tar]['probeless_all'], label=f'Probeless, {src if src != "rest" else "restaurant"} {rightarrow} {tar}', color=color, linestyle='solid', marker='^', markevery=50, markersize=6)
            plt.plot(range(768), res[src][tar]['linear_all'], label=f'Linear, {src if src != "rest" else "restaurant"} {rightarrow} {tar}', color=color, linestyle='dashed', marker='s', markevery=50, markersize=6)
    plt.axhline(0., color='gray', linestyle='dotted')
    plt.legend(loc=(0.45, 0.1), prop={'size': 10})
    plt.xlabel('k', fontsize=16)
    plt.ylabel('\u0394', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.tight_layout()
    plt.savefig(Path('results', f'beta={str(beta)}'))
    plt.close()
    # # with open(Path(root_path, 'comparison.txt'), 'w+') as f:
    # #     sys.stdout = f
    # #     print(res)


def plot_transpose(all_betas, k=50):
    res = {'probeless': [], 'linear': [], 'init': [], 'UB': []}
    tasks = ['sentiment', 'aspect']
    res = {'airline': {'dvd': {}}, 'rest': {'service': {}}}
    for task in tasks:
        finetuned_accs = finetuned[task]['all']
        source = 'airline' if task == 'sentiment' else 'rest'
        target = 'dvd' if task == 'sentiment' else 'service'
        res[source][target]['probeless_all'] = []
        res[source][target]['linear_all'] = []
        root_path = Path('results', task, source, target, 'test', 'all')
        for beta in range(1, 11):
            probeless_res, linear_res, init_res, ub_res = [], [], [], []
            for seed in range(1, 6):
                seed_dir = Path(root_path, f'seed_{seed}')
                probeless_name = 'translation_all_betas_probeless.txt' if all_betas else \
                    f'translation_{str(beta)}_probeless.txt'
                probeless_path = Path(seed_dir, probeless_name)
                # probeless_path = Path(seed_dir, f'translation_{str(beta)}_probeless.txt')
                curr_init, curr_probeless_acc, curr_upper_bound, _ = deltas(probeless_path, finetuned_accs[target][seed - 1], all_betas, beta)
                probeless_res.append(curr_probeless_acc[k])
                init_res.append(curr_init)
                ub_res.append(curr_upper_bound)
                linear_name = 'translation_all_betas_linear.txt' if all_betas else \
                    f'translation_{str(beta)}_linear.txt'
                linear_path = Path(seed_dir, linear_name)
                # linear_path = Path(seed_dir, f'translation_{str(beta)}_linear.txt')
                curr_linear_acc = deltas(linear_path, finetuned_accs[target][seed - 1], all_betas, beta)[1]
                linear_res.append(curr_linear_acc[k])
            probeless_mean = np.mean(probeless_res, axis=0)
            probeless_std = np.std(probeless_res, axis=0)
            linear_mean = np.mean(linear_res, axis=0)
            linear_std = np.std(linear_res, axis=0)
            # res['probeless'].append((round(probeless_mean, 1), round(probeless_std, 1)))
            # res['linear'].append((round(linear_mean, 1), round(linear_std, 1)))
            res[source][target]['probeless_all'].append(probeless_mean)
            res[source][target]['linear_all'].append(linear_mean)
    rightarrow = u"\u2192"
    for src in res.keys():
        for tar in res[src].keys():
            # linestyle = 'solid' if src != 'airline' else 'dashed'
            color = 'C0' if src != 'airline' else 'C1'
            plt.plot(range(1, 11), res[src][tar]['probeless_all'],
                     label=f'Probeless, {src if src != "rest" else "restaurant"} {rightarrow} {tar}', color=color,
                     linestyle='solid', marker='^', markersize=4)
            plt.plot(range(1, 11), res[src][tar]['linear_all'],
                     label=f'Linear, {src if src != "rest" else "restaurant"} {rightarrow} {tar}', color=color,
                     linestyle='dashed', marker='s', markersize=4)
    plt.axhline(0., color='gray', linestyle='dotted')
    plt.legend(prop={'size': 10})
    plt.xlabel('\u03B2', fontsize=16)
    plt.ylabel('\u0394', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.tight_layout()
    plt.savefig(Path('results', f'k={str(k)}'))
    plt.close()


if __name__ == "__main__":
    # task = 'sentiment'
    one_file = True
    ens = False
    # probeless_vs_linear(one_file, ens)
    plot_transpose(one_file)
    # res = run_task(task)



