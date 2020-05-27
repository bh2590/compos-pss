import ipdb as pdb
from .util import flatten

import torch
from torch import nn
from torch import optim
from nltk import Tree
from typing import List, Dict, Tuple, Optional, Union, Callable
import numpy as np
from tqdm import tqdm

import constants


class L1Dist(nn.Module):
    def forward(self, pred, target):
        return torch.abs(pred - target).sum()


class CosDist(nn.Module):
    def forward(self, x, y):
        nx, ny = nn.functional.normalize(x), nn.functional.normalize(y)
        return 1 - (nx * ny).sum()


class Objective(nn.Module):
    def __init__(self, vocab_lookup: Dict[str, int], repr_size: int, comp_fn: Callable, err_fn: Callable,
                 zero_init: bool,
                 init_embedding: Optional[np.ndarray] = None):
        super().__init__()
        self.vocab_lookup = vocab_lookup
        if constants.UNK_TOKEN not in self.vocab_lookup:
            self.vocab_lookup[constants.UNK_TOKEN] = len(vocab_lookup)
        self.vocab_size = len(vocab_lookup)
        self.embed_dim = repr_size
        # self.emb = nn.Embedding(len(vocab), repr_size)
        self.emb = self.define_embedding(init_embedding)
        if zero_init:
            self.emb.weight.data.zero_()
        self.comp_fn = comp_fn
        self.err_fn = err_fn

    def index(self, e: str) -> torch.Tensor:
        i = self.vocab_lookup.get(e, self.vocab_lookup[constants.UNK_TOKEN])
        ind = torch.LongTensor([i])
        return ind

    def define_embedding(self, embedding: Optional[np.ndarray] = None) -> torch.nn.modules.sparse.Embedding:
        if embedding is None:
            return nn.Embedding(self.vocab_size, self.embed_dim)
        else:
            embedding = torch.tensor(embedding, dtype=torch.float)
            return nn.Embedding.from_pretrained(embedding)

    def compose(self, e: Union[Tuple, str]) -> torch.Tensor:
        if isinstance(e, tuple):
            # pdb.set_trace()
            ret_tup = (self.compose(ee) for ee in e)
            return self.comp_fn(*ret_tup)
        return self.emb(self.index(e))

    def interpret(self, subtree: Union[Tree, str]) -> torch.Tensor:
        # Terminal nodes are str:
        if isinstance(subtree, str):
            i = self.vocab_lookup.get(subtree, self.vocab_lookup[constants.UNK_TOKEN])
            ind = torch.tensor([i], dtype=torch.long)
            return self.emb(ind)
        # Non-branching nodes:
        elif len(subtree) == 1:
            return self.interpret(subtree[0])
        # Branching nodes:
        else:
            left_subtree, right_subtree = subtree[0], subtree[1]
            left_subtree = self.interpret(left_subtree)
            right_subtree = self.interpret(right_subtree)
            combined_rep = self.comp_fn(left_subtree, right_subtree)
            # root_rep = self.hidden_activation(self.tree_layer(combined))
            return combined_rep

    def forward(self, rep: torch.Tensor, expr: Union[Tree, Tuple, torch.Tensor, str]) -> torch.Tensor:
        # pdb.set_trace()
        if isinstance(expr, torch.Tensor) or isinstance(expr, Tuple) or isinstance(expr, str):
            compositional_rep = self.compose(expr)
        elif isinstance(expr, Tree):
            compositional_rep = self.interpret(expr)
        else:
            raise TypeError("expr input can only be one of nltk.Tree or Tuple, {} type given".format(type(expr)))
        return self.err_fn(compositional_rep, rep)


def evaluate(reps, exprs, comp_fn, err_fn, quiet=False, steps=400, include_pred=False, zero_init=True):
    pdb.set_trace()
    vocab = {}
    for expr in exprs:
        toks = flatten(expr)
        for tok in toks:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    treps = [torch.FloatTensor([r]) for r in reps]
    texprs = exprs

    obj = Objective(vocab, reps[0].size, comp_fn, err_fn, zero_init)
    opt = optim.RMSprop(obj.parameters(), lr=0.01)

    pdb.set_trace()
    with tqdm(total=steps) as pbar:
        for t in range(steps):
            opt.zero_grad()
            # errs = []
            # for r, e in zip(treps, texprs):
            #     tmp_err = obj(r, e)
            #     errs.append(tmp_err)
            errs = [obj(r, e) for r, e in zip(treps, texprs)]
            loss = sum(errs)
            loss.backward()
            if not quiet and t % 100 == 0:
                print(loss.item())
            opt.step()
            pbar.update(1)

    # for r, e in zip(treps, texprs):
    #    print(r, obj.compose(e))
    # assert False
    pdb.set_trace()
    final_errs = [err.item() for err in errs]
    if include_pred:
        lexicon = {
            k: obj.emb(torch.LongTensor([v])).data.cpu().numpy()
            for k, v in vocab.items()
        }
        composed = [obj.compose(t) for t in texprs]
        return final_errs, lexicon, composed
    else:
        return final_errs
