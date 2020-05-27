import os, sys

import sst
from typing import List, Dict, Union, NamedTuple
from nltk import Tree
import numpy as np
from collections import defaultdict
import ipdb as pdb
from ipdb import slaunch_ipdb_on_exception
import json

from base_classes import BaseDataset
from configs import parser, tre_score_gen_config

# =============================================================================
# CONSTANT Args
# =============================================================================

SST_DATA_HOME = 'sst'
TRE_DATA_HOME = os.path.join("tre", "rep2_data")


# =============================================================================
# Class Defs
# =============================================================================

class TreOutput(object):
    def __init__(self, text: str, score: float):
        self.text = text
        self.score = score

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        return cls(**json_dict)

    def __eq__(self, other):
        return self.text == other.text and self.score == other.score


class TREScoring(object):
    def __init__(self, **kwargs):
        pdb.set_trace()
        self.compositionality_type = kwargs['compositionality_type']
        self.root_weight = 5.0

    def get_compositionality(self, tree: Tree) -> float:
        if self.compositionality_type == 'tree_impurity':
            return self.calc_tree_impurity(tree)
        elif self.compositionality_type == 'node_switching':
            return self.calc_node_switching(tree)
        else:
            raise ValueError("Unsupported compositionality_type: Either one from tree_impurity or node_switching")

    def calc_tree_impurity(self, tree: Tree) -> float:
        ddict = defaultdict(list)

        def traverse_tree(tree: Union[Tree, str]):
            ddict['text'].append(" ".join(tree.leaves()))  # TODO: For debugging purposes only, can remove later
            ddict['label'].append(float(tree.label()))
            for subtree in tree:
                if isinstance(subtree, Tree):
                    traverse_tree(subtree)

        traverse_tree(tree)
        tree_impurity = np.abs(np.mean(ddict['label']) - float(tree.label()))
        return tree_impurity

    def calc_node_switching(self, tree: Tree) -> float:
        ddict = defaultdict(list)

        def traverse_tree(tree: Union[Tree, str], weight: float):
            # weight = max(weight, 1)
            if isinstance(tree[0], str):
                return
            tree_label = float(tree.label())
            leftsubtree, rightsubtree = tree[0], tree[1]
            left_label, right_label = float(leftsubtree.label()), float(rightsubtree.label())
            avg_child_label = (left_label + right_label) / 2.0
            label_switch = abs(tree_label - avg_child_label) * weight
            ddict['text'].append(" ".join(tree.leaves()))  # TODO: For debugging purposes only, can remove later
            ddict['label'].append(tree.label())  # TODO: For debugging purposes only, can remove later
            ddict['label_switch'].append(label_switch)
            traverse_tree(leftsubtree, weight / 2.)
            traverse_tree(rightsubtree, weight / 2.)

        traverse_tree(tree, self.root_weight)
        mean_label_switching = np.mean(ddict['label_switch'])
        return mean_label_switching

    def __call__(self, dataset: BaseDataset) -> Dict[str, TreOutput]:
        pdb.set_trace()
        ret_dict = dict()
        for chnk_num, (keys, trees) in enumerate(dataset):
            for i in range(len(keys)):
                compositional_score = self.get_compositionality(trees[i])
                tre_obj = TreOutput(" ".join(trees[i].leaves()), compositional_score)
                ret_dict[keys[i]] = tre_obj
        return ret_dict


class SSTTREScoringDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.key_prefix == 'train':
            self.sst_reader = sst.train_reader(SST_DATA_HOME)
        elif self.key_prefix == 'dev':
            self.sst_reader = sst.dev_reader(SST_DATA_HOME)
        else:
            raise ValueError("Invalid key-prefix for SST reader")

    def load_data(self) -> None:
        """
        Load SST data
        :return: Nothing just set self.data
        """
        pdb.set_trace()
        self.data = [t for t, _ in self.sst_reader]


def dumper(obj):
    try:
        return obj.to_json()
    except:
        return obj.__dict__


def run_ground_truth_comp_scoring(kwargs: Dict):
    pdb.set_trace()
    dataset_dict = {
        'sst': SSTTREScoringDataset,
    }

    dataset = dataset_dict[kwargs['dataset_name']](**kwargs)
    dataset.load_data()

    compositionality_scoring_pipeline = TREScoring(**kwargs)
    compositionality_scores_dict = compositionality_scoring_pipeline(dataset)
    prefix = '_'.join([kwargs['dataset_name'], kwargs['compositionality_type'], ''])
    pdb.set_trace()
    scored_fname = prefix + kwargs['comp_score_file']
    with open(scored_fname, 'w') as f:
        json.dump(compositionality_scores_dict, f, indent=4, default=dumper)
    check_dict = get_scored_dict(scored_fname)
    assert check_dict == compositionality_scores_dict, "JSON Encoder/Decoder consistency error"


def get_scored_dict(score_file: str) -> Dict[str, TreOutput]:
    pdb.set_trace()
    with open(score_file, 'r') as f:
        serialized_data_dict = json.loads(f.read())
    tre_scored_dict = dict()
    for k, v in serialized_data_dict.items():
        tre_scored_dict[k] = TreOutput.from_json(v)
    return tre_scored_dict


if __name__ == "__main__":
    parser = tre_score_gen_config(parser)
    args = parser.parse_args()

    if not args.debug:
        pdb.set_trace = lambda: None
    with slaunch_ipdb_on_exception():
        run_ground_truth_comp_scoring(vars(args))
