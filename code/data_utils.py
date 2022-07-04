import pickle
import h5py
from pathlib import Path
import numpy as np

sentiment_data = {}
sentiment_data['train_paths'] = {'airline': 'data/blitzer_data/airline/train',
                                 'books': 'data/blitzer_data/books/train',
                                 'dvd': 'data/blitzer_data/dvd/train',
                                 'electronics': 'data/blitzer_data/electronics/train',
                                 'kitchen': 'data/blitzer_data/kitchen/train'}

sentiment_data['dev_paths'] = {'airline': 'data/blitzer_data/airline/dev',
                               'books': 'data/blitzer_data/books/dev',
                               'dvd': 'data/blitzer_data/dvd/dev',
                               'electronics': 'data/blitzer_data/electronics/dev',
                               'kitchen': 'data/blitzer_data/kitchen/dev'}

sentiment_data['test_paths'] = {'airline': 'data/blitzer_data/airline/test',
                                'books': 'data/blitzer_data/books/test',
                                'dvd': 'data/blitzer_data/dvd/test',
                                'electronics': 'data/blitzer_data/electronics/test',
                                'kitchen': 'data/blitzer_data/kitchen/test'}

rumour_data = {}
rumour_data['train_paths'] = {'charliehebdo': 'data/rumour_data/charliehebdo/train',
                              'ferguson': 'data/rumour_data/ferguson/train',
                              'germanwings-crash': 'data/rumour_data/germanwings-crash/train',
                              'ottawashooting': 'data/rumour_data/ottawashooting/train',
                              'sydneysiege': 'data/rumour_data/sydneysiege/train'}

rumour_data['dev_paths'] = {'charliehebdo': 'data/rumour_data/charliehebdo/dev',
                            'ferguson': 'data/rumour_data/ferguson/dev',
                            'germanwings-crash': 'data/rumour_data/germanwings-crash/dev',
                            'ottawashooting': 'data/rumour_data/ottawashooting/dev',
                            'sydneysiege': 'data/rumour_data/sydneysiege/dev'}

rumour_data['test_paths'] = {'charliehebdo': 'data/rumour_data/charliehebdo/test',
                             'ferguson': 'data/rumour_data/ferguson/test',
                             'germanwings-crash': 'data/rumour_data/germanwings-crash/test',
                             'ottawashooting': 'data/rumour_data/ottawashooting/test',
                             'sydneysiege': 'data/rumour_data/sydneysiege/test'}

mnli_data = {}
mnli_data['train_paths'] = {'fiction': 'data/mnli_data/fiction/train',
                            'government': 'data/mnli_data/government/train',
                            'slate': 'data/mnli_data/slate/train',
                            'telephone': 'data/mnli_data/telephone/train',
                            'travel': 'data/mnli_data/travel/train'}

mnli_data['dev_paths'] = {'fiction': 'data/mnli_data/fiction/dev',
                          'government': 'data/mnli_data/government/dev',
                          'slate': 'data/mnli_data/slate/dev',
                          'telephone': 'data/mnli_data/telephone/dev',
                          'travel': 'data/mnli_data/travel/dev'}

mnli_data['test_paths'] = {'fiction': 'data/mnli_data/fiction/test',
                           'government': 'data/mnli_data/government/test',
                           'slate': 'data/mnli_data/slate/test',
                           'telephone': 'data/mnli_data/telephone/test',
                           'travel': 'data/mnli_data/travel/test'}

aspect_data = {}
aspect_data['train_paths'] = {'device': 'data/absa_data/device/train',
                              'laptops': 'data/absa_data/laptops/train',
                              'rest': 'data/absa_data/rest/train',
                              'service': 'data/absa_data/service/train'}

aspect_data['dev_paths'] = {'device': 'data/absa_data/device/dev',
                            'laptops': 'data/absa_data/laptops/dev',
                            'rest': 'data/absa_data/rest/dev',
                            'service': 'data/absa_data/service/dev'}

aspect_data['test_paths'] = {'device': 'data/absa_data/device/test',
                             'laptops': 'data/absa_data/laptops/test',
                             'rest': 'data/absa_data/rest/test',
                             'service': 'data/absa_data/service/test'}

data_dict = {'sentiment': sentiment_data,
             'rumour': rumour_data,
             'mnli': mnli_data,
             'aspect': aspect_data}

data_dirs = {'sentiment': 'blitzer_data',
             'rumour': 'rumour_data',
             'mnli': 'mnli_data',
             'aspect': 'absa_data'}

aspect_label_mapping = {'O': 0, 'B-AS': 1, 'I-AS': 1}


def load_task_data(task, dev_domain, source_domains, test, data_size, seed):
    set_str = 'test' if test else 'dev'
    pkl_path = Path('idani-models', task, f'{source_domains[0]}')
    data_dir = data_dirs[task]
    dir_name = 'all' if data_size == -1 else str(data_size)
    seed_str = f'seed_{seed}'
    X_train_domain, X_train_task, Y_train_domain, Y_train_task, X_test_domain, X_test_task, Y_test_domain, Y_test_task \
        = [], [], [], [], [], [], [], []
    for i, d in enumerate(source_domains):
        if task in ['mnli', 'aspect']:
            f = h5py.File(Path(pkl_path, d, dir_name, seed_str, 'train_parsed.pkl'), 'r')
            domain_data = np.array(f['representations']), np.array(f['labels'])
            X_train_domain.append(domain_data[0])
            X_train_task.append(domain_data[0])
            f.close()
        else:
            with open(Path(pkl_path, d, dir_name, seed_str, 'train_parsed.pkl'), 'rb') as f:
                domain_data = pickle.load(f)
            X_train_domain.extend([t for t in domain_data[0]])
            X_train_task.extend([t for t in domain_data[0]])
        Y_train_domain.extend([0] * len(domain_data[1]))
        Y_train_task.extend(domain_data[1])
        if task in ['mnli', 'aspect']:
            f = h5py.File(Path(pkl_path, d, dir_name, seed_str, f'{set_str}_parsed.pkl'), 'r')
            domain_data = np.array(f['representations']), np.array(f['labels'])
            X_test_domain.append(domain_data[0])
            # X_test_task.append(domain_data[0])
            f.close()
        else:
            with open(Path(pkl_path, d, dir_name, seed_str, f'{set_str}_parsed.pkl'), 'rb') as f:
                domain_data = pickle.load(f)
            X_test_domain.extend([t for t in domain_data[0]])
            # X_test_task.extend([t for t in domain_data[0]])
        Y_test_domain.extend([0] * len(domain_data[1]))
        # Y_test_task.extend(domain_data[1])
    if task in ['mnli', 'aspect']:
        f = h5py.File(Path(pkl_path, dev_domain, dir_name, seed_str, 'train_parsed.pkl'), 'r')
        domain_data = np.array(f['representations']), np.array(f['labels'])
        X_train_domain.append(domain_data[0])
        Y_train_domain.extend([1] * len(domain_data[1]))
        f.close()
    else:
        with open(Path(pkl_path, dev_domain, dir_name, seed_str, 'train_parsed.pkl'), 'rb') as f:
            domain_data = pickle.load(f)
        X_train_domain.extend([t for t in domain_data[0]])
        Y_train_domain.extend([1] * len(domain_data[1]))
    if task in ['mnli', 'aspect']:
        f = h5py.File(Path(pkl_path, dev_domain, dir_name, seed_str, f'{set_str}_parsed.pkl'), 'r')
        domain_data = np.array(f['representations']), np.array(f['labels'])
        X_test_domain.append(domain_data[0])
        Y_test_domain.extend([1] * len(domain_data[1]))
        X_test_task.extend([domain_data[0]])
        Y_test_task.extend(domain_data[1])
        f.close()
    else:
        with open(Path(pkl_path, dev_domain, dir_name, seed_str, f'{set_str}_parsed.pkl'), 'rb') as f:
            domain_data = pickle.load(f)
        X_test_domain.extend([t for t in domain_data[0]])
        Y_test_domain.extend([1] * len(domain_data[1]))
        X_test_task.extend([t for t in domain_data[0]])
        Y_test_task.extend(domain_data[1])
    X_train_domain = np.array(X_train_domain) if task not in ['mnli', 'aspect'] else np.concatenate(X_train_domain)
    X_train_task = np.array(X_train_task) if task not in ['mnli', 'aspect'] else np.concatenate(X_train_task)
    Y_train_domain = np.array(Y_train_domain)
    X_test_domain = np.array(X_test_domain) if task not in ['mnli', 'aspect'] else np.concatenate(X_test_domain)
    X_test_task = np.array(X_test_task) if task not in ['mnli', 'aspect'] else np.concatenate(X_test_task)
    Y_test_domain = np.array(Y_test_domain)
    return X_train_domain, X_train_task, Y_train_domain, Y_train_task, X_test_domain, X_test_task, Y_test_domain, Y_test_task


def load_domain_data(task, dev_domain, test_domain):
    pkl_path = Path('idani-models', task, f'{dev_domain}_{test_domain}')
    data_dir = data_dirs[task]
    relevant_domains = [d.name for d in Path('data', data_dir).glob('*') if d.is_dir() and d.name != test_domain]
    train_domains = [d for d in relevant_domains if d != dev_domain]
    X_train, Y_train, X_dev, Y_dev = [], [], [], []
    for i, d in enumerate(relevant_domains):
        if task in ['mnli', 'aspect']:
            f = h5py.File(Path(pkl_path, d, 'train_parsed.pkl'), 'r')
            domain_data = np.array(f['representations']), np.array(f['labels'])
            X_train.append(domain_data[0])
            f.close()
        else:
            with open(Path(pkl_path, d, 'train_parsed.pkl'), 'rb') as f:
                domain_data = pickle.load(f)
            X_train.extend([t for t in domain_data[0]])
        Y_train.extend([1 if d == dev_domain else 0] * len(domain_data[1]))
        if task in ['mnli', 'aspect']:
            f = h5py.File(Path(pkl_path, d, 'dev_parsed.pkl'), 'r')
            domain_data = np.array(f['representations']), np.array(f['labels'])
            X_dev.append(domain_data[0])
            f.close()
        else:
            with open(Path(pkl_path, d, 'dev_parsed.pkl'), 'rb') as f:
                domain_data = pickle.load(f)
            X_dev.extend([t for t in domain_data[0]])
        Y_dev.extend([1 if d == dev_domain else 0] * len(domain_data[1]))
    X_train = np.concatenate(X_train) if task in ['mnli', 'aspect'] else np.array(X_train)
    Y_train = np.array(Y_train)
    X_dev = np.concatenate(X_dev) if task in ['mnli', 'aspect'] else np.array(X_dev)
    Y_dev = np.array(Y_dev, dtype=int)
    majority = np.bincount(Y_dev).max() / Y_dev.shape[0]
    print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)
    return X_train, Y_train, X_dev, Y_dev, majority
