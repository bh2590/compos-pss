from nltk import Tree
from nltk.tokenize import word_tokenize
from typing import List, Tuple, Union


class Logger(object):
    def __init__(self, keys, formats, width=12):
        self.keys = keys
        self.formats = formats
        self.data = {}
        self.width = width

    def begin(self):
        print('| ' + ' | '.join(('%%%ds' % self.width) % k for k in self.keys) + ' |')

    def update(self, key, value):
        if key not in self.keys:
            return
        assert key in self.keys
        assert key not in self.data
        self.data[key] = value

    def print(self):
        out = []
        for k, f in zip(self.keys, self.formats):
            if k in self.data:
                out.append(('%%%s%s' % (self.width, f)) % self.data[k])
            else:
                out.append(' ' * self.width)
        print('| ' + ' | '.join(out) + ' |')
        self.data.clear()


def flatten_tree(tree: Tree):
    return word_tokenize(" ".join(tree.leaves()).lower())


def flatten_phrase(l: Union[Tuple, str]):
    if not isinstance(l, tuple):
        return (l,)

    out = ()
    for ll in l:
        out = out + flatten(ll)
    return out


def flatten(l: Union[Tuple, Tree, str]):
    if isinstance(l, str) or isinstance(l, tuple):
        return flatten_phrase(l)
    elif isinstance(l, Tree):
        return flatten_tree(l)
    else:
        TypeError("Invalid type to flatten")
