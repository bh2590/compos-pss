from datasets import SSTDataset, PhraseDataset
from configs import parser, tre_config
import ipdb as pdb
from ipdb import slaunch_ipdb_on_exception
from typing import Dict
import json
from tre import evals2 as tre_eval
import torch.nn as nn
import constants
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
from nltk import Tree

sns.set(font_scale=1.5)
sns.set_style("ticks", {'font.family': 'serif'})
import os


def ranks(arr):
    return np.array(arr).argsort().argsort()


def main(kwargs: Dict):
    pdb.set_trace()

    composition_fn = constants.composition_fn_dict[kwargs['composition_fn']]
    dataset = constants.dataset_dict[kwargs['dataset_name']](**kwargs)
    # dataset.load_data()

    prefix = '_'.join([kwargs['dataset_name'], kwargs['encoder_name'], ''])
    with open(prefix + kwargs['output_embeddings_file'], 'r') as f:
        output = json.load(f)
    # metadata = output['metadata']

    # Assumes the expressions are the keys and the representations are the
    # values.
    if 'embedding_dict' in output:
        embedding_dict = output['embedding_dict']
    else:
        embedding_dict = output
    reps, keys, exprs, offset = dataset.prepare_tre_input(embedding_dict)

    # Assumes the compositional expressions and representations are the last
    # n items.
    errs = tre_eval.evaluate(
        np.array(reps), exprs, composition_fn(), tre_eval.CosDist(),
        zero_init=False)

    if isinstance(exprs[0], Tree):
        exprs = [" ".join(e.leaves()) for e in exprs]
    ecomp = exprs[-offset:]
    errs = errs[-offset:]

    pdb.set_trace()
    scores = dataset.get_gt_tre_scores(keys[-offset:])

    r_errs = ranks(errs)
    r_scores = ranks(scores)

    data = pd.DataFrame({'err': r_errs, 'score': r_scores})
    sns.lmplot(x='err', y='score', data=data)
    plt.title('Encoder: {encoder_name}; Dataset: {dataset_name}'.format(
        encoder_name=kwargs['encoder_name'], dataset_name=kwargs['dataset_name']))
    plt.xlabel('TRE (rank)')
    plt.ylabel('compositionality (rank)')
    # plt.ylim(0, 5)
    plt.savefig('%s_correl_plot_full.png' % prefix, format='png')
    # plt.show()
    print(scipy.stats.spearmanr(errs, scores))

    comb = zip(scores, errs, ecomp)
    comb = sorted(comb, key=lambda x: x[1])
    df = pd.DataFrame(comb, columns=["human score", "model err", "words"])
    df['model_ranks'] = ranks(df['model err'].values)
    df['human_ranks'] = ranks(df['human score'].values)
    df['rank_correl'] = scipy.stats.spearmanr(df['model err'].values, df['human score'].values).correlation
    df.to_csv('%s_compositional_scores_full.csv' % prefix, index=True)

    pdb.set_trace()
    print("compositional:")
    print("%20s %20s %40s" % ("human score", "model err", "words"))
    for c in comb[:5]:
        if isinstance(c[2], Tree):
            text = " ".join(c[2].leaves())
            print("%20.2f %20.2f %40s" % (c[0], c[1], text))
        else:
            print("%20.2f %20.2f %40s" % c)
    print()
    print("non-compositional:")
    print("%20s %20s %40s" % ("human score", "model err", "words"))
    for c in comb[-5:]:
        if isinstance(c[2], Tree):
            text = " ".join(c[2].leaves())
            print("%20.2f %20.2f %40s" % (c[0], c[1], text))
        else:
            print("%20.2f %20.2f %40s" % c)


if __name__ == "__main__":
    parser = tre_config(parser)
    args = parser.parse_args()

    if not args.debug:
        pdb.set_trace = lambda: None

    with slaunch_ipdb_on_exception():
        main(vars(args))
