# idani
[IDANI: Inference-time Domain Adaptation via Neuron-level Interventions](https://arxiv.org/abs/2206.00259)

## Setup

Install the required libraries by running `pip install -r requirements.txt`.

## Experiments

1. Run `python code/fineTuning.py -task TASK -src_domain SRC -seed SEED`,
where TASK should be one of `['sentiment', 'mnli', 'aspect']`, SRC is the source domain (refer to the paper or `data_utils.py` for relevant domains for each task), and SEED is an integer random seed. This will fine tune a pre-trained BERT model on the source domain and save it.
2. Run `python code/parsing.py -task TASK -src_domain SRC -seed SEED`. This will process data from all domains through the model that was fine tuned on the SRC domain and save the given representations.
3. Run `python code/IDANI.py -task TASK -src_domain SRC -tar_domain TAR -seed SEED -ranking_method RANKING -beta BETA`, 
where RANKING should be one of `['probeless', 'linear']`, and BETA is a hyperparameter for the algorithm. This will run the IDANI algorithm on a model that was fine tuned on the SRC domain and is facing the TAR domain during inference. The command would apply the algorithm for all `k` (num of modified neurons) in the range `[0,768]` and save a text file with the results. If `BETA == 0`, the algorithm would run for all beta values in `[1,10]`.

## Citation
If you find this repository useful in your work, please cite our paper:

```
@inproceedings{antverg-ben-david-and-yonatan-belinkov-2022-idani,
    title = "{IDANI}: Inference-time Domain Adaptation via Neuron-level Interventions",
    author = "Antverg, Omer  and
      Ben-David and Yonatan Belinkov, Eyal",
    booktitle = "Proceedings of the Third Workshop on Deep Learning for Low-Resource Natural Language Processing",
    month = jul,
    year = "2022",
    address = "Hybrid",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.deeplo-1.3",
    pages = "21--29",
    abstract = "t",
}

```
