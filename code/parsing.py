from argparse import ArgumentParser
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification
from tqdm import tqdm
import pickle
import torch
from pathlib import Path
import h5py
import numpy as np
from data_utils import data_dict, aspect_label_mapping, data_dirs

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('-task', type=str)
    argparser.add_argument('-src_domain', type=str)
    argparser.add_argument('-data_size', type=int, default=-1)
    argparser.add_argument('-seed', type=int)
    args = argparser.parse_args()
    task = args.task
    data = data_dict[task]
    src_domain = args.src_domain
    data_size = args.data_size
    seed = args.seed
    mean_word_rep = False
    print(f'task: {task}')
    print(f'src domain: {src_domain}')
    print(f'data size: {data_size}')
    print(f'seed: {seed}')
    torch.manual_seed(seed)
    source_domains = [src_domain]
    pickles_root = Path('idani-models', task, f'{source_domains[0]}')
    if not pickles_root.exists():
        pickles_root.mkdir(parents=True)
    data_dir = data_dirs[task]
    all_domains = [d.name for d in Path('data', data_dir).glob('*') if d.is_dir() and '_' not in d.name]
    bert_name = 'bert-base-cased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    for domain in all_domains:
        print(f'curr domain: {domain}')
        train_path = data['train_paths'][domain]
        dev_path = data['dev_paths'][domain]
        test_path = data['test_paths'][domain]
        curr_pkl_root = Path(pickles_root, domain, 'all' if data_size == -1 else str(data_size), f'seed_{seed}')
        if not curr_pkl_root.exists():
            curr_pkl_root.mkdir(parents=True)
        train_dump_path = Path(curr_pkl_root, 'train_parsed.pkl')
        dev_dump_path = Path(curr_pkl_root, 'dev_parsed.pkl')
        test_dump_path = Path(curr_pkl_root, 'test_parsed.pkl')
        model_dir_name = 'ft_models' if data_size == -1 else f'ft_models_{data_size}'
        saved_models_path = Path(pickles_root, model_dir_name, f'seed_{seed}')
        models_idx = [int(d.name[len('checkpoint-'):]) for d in saved_models_path.glob('*')]
        max_idx = max(models_idx)
        model_path = Path(saved_models_path, f'checkpoint-{str(max_idx)}')

        for file_path, dump_path in zip(
                [train_path, dev_path, test_path],
                [train_dump_path, dev_dump_path, test_dump_path]):

            print(f"Processing {file_path}...")
            with open(file_path, 'rb') as f:
                curr_data = pickle.load(f)
            num_labels = len(set(curr_data[1])) if task != 'aspect' else 2
                # len(set([label for labels in curr_data[1] for label in labels]))
            if task == 'aspect':
                model = BertForTokenClassification.from_pretrained(model_path.__str__(), num_labels=num_labels).to(
                    device)
            else:
                model = BertForSequenceClassification.from_pretrained(model_path.__str__(), num_labels=num_labels).to(
                    device)
            # Prepare to compute BERT embeddings
            representations = []
            labels = []
            model.eval()

            for sentence, label in tqdm(zip(curr_data[0], curr_data[1])):
                with torch.no_grad():
                    if task == 'aspect':
                        new_labels = [-100]  # ignore index for CLS token
                        for word, l in zip(sentence, label):
                            word_labels = [aspect_label_mapping[l]]
                            subtokens = tokenizer(word).data['input_ids']
                            if len(subtokens) > 3:
                                # word got split - label sub-tokens with ignore index
                                word_labels.extend([-100] * (len(subtokens) - 3))
                            new_labels.extend(word_labels)
                        sentence = ' '.join(sentence)
                        padding_num = 256 - len(new_labels)
                        new_labels.extend([-100] * padding_num)
                        label = new_labels
                    if task == 'mnli':
                        tokens = tokenizer(sentence[0], sentence[1], padding='max_length', truncation=True, max_length=256,
                                           return_tensors="pt").to(device).data
                    else:
                        tokens = tokenizer(sentence, padding='max_length', truncation=True, max_length=256,
                                       return_tensors="pt").to(device).data
                    outputs = \
                    model(tokens['input_ids'], attention_mask=tokens['attention_mask'],
                          token_type_ids=tokens["token_type_ids"], output_hidden_states=True)[1][-1][0]
                    if task == 'aspect':
                        # omit CLS and PAD reps and labels
                        relevant_outputs = outputs[1:-padding_num]
                        relevant_labels = label[1:-padding_num]
                        new_outputs, new_labels = [], []
                        for i, (r, l) in enumerate(zip(relevant_outputs, relevant_labels)):
                            if l == -100:
                                # ignore the token
                                continue
                            if i < len(relevant_labels) and relevant_labels[i] != -100:
                                # word did not split
                                new_outputs.append(r.cpu().numpy())
                                new_labels.append(l if l == 0 else 1)
                                continue
                            # word was sub-tokenized - compute mean representation for all its sub-tokens
                            if mean_word_rep:
                                start_idx, end_idx = i, i + 1
                                while len(relevant_labels) > end_idx and relevant_labels[end_idx] == -100:
                                    end_idx += 1
                                word_rep = relevant_outputs[start_idx:end_idx].mean(dim=0).cpu().numpy()
                                new_outputs.append(word_rep)
                                new_labels.append(l if l == 0 else 1)
                        representations.append(new_outputs)
                        labels.extend(new_labels)
                    else:
                        representations.append(outputs[0].cpu().numpy())
                        labels.append(label)
            representations = np.concatenate(representations) if task == 'aspect' else np.array(representations)
            labels = np.array(labels, dtype=float)
            # Save final results
            if task in ['mnli', 'aspect']:
                hf = h5py.File(dump_path, 'w')
                hf.create_dataset('representations', data=representations)
                hf.create_dataset('labels', data=labels)
                hf.close()
            else:
                with open(dump_path, "wb+") as h:
                    pickle.dump((representations, labels), h)
