import os, sys
import sst
from typing import List, Dict, Union, Tuple
from copy import deepcopy
from numpy import ndarray
from nltk import Tree
import ipdb as pdb
from base_classes import BaseDataset

SST_DATA_HOME = 'sst'
TRE_DATA_HOME = os.path.join("tre", "rep2_data")


class SSTDataset(BaseDataset):
    def __init__(self, **kwargs):
        pdb.set_trace()
        kwargs['keyval_data'] = False
        super().__init__(**kwargs)
        self.initialize_sst_reader()

    def initialize_sst_reader(self):
        if self.key_prefix == 'train':
            self.sst_reader = sst.train_reader(SST_DATA_HOME, class_func=sst.ternary_class_func)
        elif self.key_prefix == 'dev':
            self.sst_reader = sst.dev_reader(SST_DATA_HOME, class_func=sst.ternary_class_func)
        else:
            raise ValueError("Invalid key-prefix for SST reader")

    def load_data(self) -> None:
        """
        Load SST data
        :return: Nothing just set self.data
        """
        pdb.set_trace()
        sst_text_list = [(" ".join(t.leaves()), label) for t, label in self.sst_reader]
        X, _ = zip(*sst_text_list)
        self.data = list(X)
        self.keys = [self.key_prefix + str(i) for i in range(len(self.data))]

    def prepare_tre_input(self, vecs: Dict[str, Union[ndarray, List[float]]]) -> Tuple[
        Union[List[ndarray], ndarray], List[str], List[Union[str, Tuple, Tree]], int]:
        pdb.set_trace()
        reps = list(vecs.values())
        trees = [t for t, _ in self.sst_reader]
        # TODO: Remove this check after a few times running
        self.load_data()
        check_text = [" ".join(t.leaves()) for t in trees]
        assert check_text == self.data, "Data unequal"
        offset = 0
        return reps[:], self.keys[:], trees[:], offset

    def get_gt_tre_scores(self, keys: List[Union[str, Tuple]]) -> List[float]:
        pdb.set_trace()
        from run_compositionality_score_gen import get_scored_dict
        # TODO: remove hardcoded filename
        all_scores = get_scored_dict("sst_node_switching_comp_scores.json")
        scores = [all_scores[k].score for k in keys]
        return scores


class PhraseDataset(BaseDataset):
    def __init__(self, **kwargs):
        pdb.set_trace()
        kwargs['keyval_data'] = True
        super().__init__(**kwargs)
        self.input_filename = os.path.join(TRE_DATA_HOME, 'reddy.txt')

    def load_data(self) -> None:
        """
        Only bigrams considered here
        :return: Nothing just set self.data
        """
        pdb.set_trace()
        unigrams = []
        bigrams = []
        self.scores = {}
        with open(self.input_filename, 'r') as f:
            next(f)  # header
            for line in f:
                line = line.split()
                w1, w2 = line[:2]
                sim = float(line[6])
                w1, _ = w1.split('-')
                w2, _ = w2.split('-')
                bigram = " ".join((w1, w2))
                unigrams.append(w1)
                unigrams.append(w2)
                bigrams.append(bigram)
                self.scores[bigram] = sim
        self.prim_len = len(unigrams)
        self.comp_len = len(bigrams)
        self.data = unigrams + bigrams
        self.keys = deepcopy(self.data)

    def prepare_tre_input(self, vecs: Dict[str, Union[ndarray, List[float]]]) -> Tuple[
        Union[List[ndarray], ndarray], List[str], List[Union[str, Tuple, Tree]], int]:
        pdb.set_trace()
        eprim = []
        rprim = []
        ecomp = []
        rcomp = []
        bigrams = [tuple(e.split()) for e in vecs.keys() if len(e.split()) > 1]
        for w1, w2 in bigrams:
            b = (w1, w2)
            key = " ".join(b)
            if not (w1 in vecs and w2 in vecs and key in vecs):
                continue
            eprim.append(w1)
            rprim.append(vecs[w1])
            eprim.append(w2)
            rprim.append(vecs[w2])
            ecomp.append(b)
            rcomp.append(vecs[key])
        exprs = eprim + ecomp
        ret_keys = [" ".join(e) if isinstance(e, tuple) else e for e in exprs]
        reps = rprim + rcomp
        offset = len(rcomp)
        pdb.set_trace()
        return reps, ret_keys, exprs, offset

    def get_gt_tre_scores(self, keys: List[Union[str, Tuple]]) -> List[float]:
        self.load_data()
        scores = [self.scores[e] for e in keys]
        return scores
