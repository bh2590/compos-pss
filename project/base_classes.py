from typing import List, Dict, Union, Type, Tuple
from numpy import array_split, ndarray
import ipdb as pdb
from nltk import Tree
import logging

logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s', level=logging.INFO)


def chunk_gen(inp_list: List, batch_size: int):
    num_chunks = len(inp_list) // batch_size + 1
    for chunk in array_split(inp_list, num_chunks):
        yield chunk


class BaseDataset(object):
    def __init__(self, chunksize: int = 10000, keyval_data: bool = False, key_prefix: str = '', **kwargs):
        """

        :param chunksize: The chunksize of the data that can be held in memory
        :param keyval_data: If the data-stream is a List of key-text Tuples or just List of texts
        :param key_prefix: The key prefix
        """
        self.key_prefix = key_prefix
        self.data = []
        self.keys = []
        self.keyval_data = keyval_data
        self.chunksize = chunksize

    def load_data(self) -> None:
        """
        Loads the data and sets the self.data attribute and (optionally) the self.keys attribute
        :return:
        """
        raise NotImplementedError("Data loading to be implemented in child class")

    def get_gt_tre_scores(self, keys: List[Union[str, Tuple]]) -> List[float]:
        """
        Return a list of the ground truth TRE/compositionality scores
        :param keys: List of keys for which we want scores
        :return: List of compositionality scores
        """
        raise NotImplementedError("TRE scores retrieval/generation to be implemented in child class")

    def prepare_tre_input(self, vecs: Dict[str, Union[ndarray, List[float]]]) -> Tuple[
        Union[List[ndarray], ndarray], List[str], List[Union[str, Tuple, Tree]], int]:
        """
        :param vecs: a dictionary of the self.keys to the sentence embedding representation
        :return: Tuple of:
                    1. the input vector representations, List of 1D arrays or single 2D array of floats
                    2. Keys
                    3. inputs (list of single word string or bigram tuple or nltk Tree form)
                    4. the offset for TRE evaluation
        """
        raise NotImplementedError("Preparing for TRE to be implemented in child class")

    def __iter__(self):
        if not self.data:
            raise ValueError("data attribute not set")

        if self.keyval_data and not self.keys:
            raise ValueError("Dataset is of key-value type but self.keys was not set")

        if not self.keys:
            logging.info("Setting default key values...")
            if not self.key_prefix:
                self.key_prefix = 'key'
            self.keys = [self.key_prefix + str(i) for i in range(len(self.data))]

        counter = 0
        key_list, text_list = [], []
        for i, (key, val) in enumerate(zip(self.keys, self.data)):
            counter += 1
            key_list.append(key)
            text_list.append(val)
            if counter == self.chunksize:
                yield key_list, text_list
                counter = 0
                key_list, text_list = [], []
        if counter > 0:
            yield key_list, text_list


class BaseEncoder(object):
    def __init__(self, **kwargs):
        self.model = None

    def load_model(self) -> None:
        """
        Loads the model and sets the self.model attribute
        :return:
        """
        raise NotImplementedError("Model loading to be implemented in child class")

    def encode(self, keys: List[str], texts: List[str]) -> Union[
        Dict[str, List], Dict[str, Dict[str, List]]]:
        raise NotImplementedError("Encoding logic to be implemented in child class")

    def __call__(self, dataset: BaseDataset) -> Union[Dict[str, List], Dict[str, Dict[str, List]]]:
        pdb.set_trace()
        if self.model is None:
            raise ValueError("model attribute must be set")
        ret_dict = dict()
        for i, (keys, texts) in enumerate(dataset):
            tmp_dict = self.encode(keys, texts)
            ret_dict.update(tmp_dict)
        return ret_dict
