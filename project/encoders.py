from bert_serving.client import BertClient
import os
from typing import List, Dict, Union
from nltk.tokenize.treebank import TreebankWordTokenizer
import numpy as np
from base_classes import BaseEncoder, chunk_gen
import ipdb as pdb
import tensorflow as tf
import tensorflow_hub as hub


import torch
from InferSent.models import InferSent

import logging

logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s', level=logging.INFO)

DEFAULT_BATCH_SIZE = 32


class BertServiceEncoder(BaseEncoder):
    def load_model(self) -> None:
        self.model = BertClient(check_length=False)

    def encode(self, keys: List[str], texts: List[str]) -> Union[
        Dict[str, List], Dict[str, Dict[str, List]]]:
        pdb.set_trace()
        bert_embeddings, bert_toks = self.model.encode(
            texts, show_tokens=True)
        bert_embeddings = [emb.tolist() for emb in bert_embeddings]
        bert_embeddings_dict = dict(zip(keys, bert_embeddings))
        return bert_embeddings_dict


class InferSentEncoder(BaseEncoder):
    def load_model(self) -> None:
        params = {'bsize': DEFAULT_BATCH_SIZE, 'word_emb_dim': 300,
                  'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0,
                  'version': 2}
        self.model = InferSent(params)
        assert os.path.exists(
            'InferSent/encoder/infersent2.pkl'), (
            'Download InferSent checkpoint here: '
            'https://dl.fbaipublicfiles.com/senteval/infersent/infersent2.pkl')
        self.model.load_state_dict(torch.load(
            'InferSent/encoder/infersent2.pkl'))
        assert os.path.exists(
            'InferSent/dataset/fastText/crawl-300d-2M-subword.vec'), (
            'Download fastText here: '
            'https://github.com/facebookresearch/InferSent#download-datasets')
        self.model.set_w2v_path(
            'InferSent/dataset/fastText/crawl-300d-2M-subword.vec')

    def encode(self, keys: List[str], texts: List[str]) -> Union[
        Dict[str, List], Dict[str, Dict[str, List]]]:
        pdb.set_trace()

        if not hasattr(self.model, 'word_vec'):
            self.model.build_vocab(texts, tokenize=True)
        else:
            self.model.update_vocab(texts)

        embeddings = self.model.encode(texts, tokenize=True)
        embeddings = [emb.tolist() for emb in embeddings]
        embeddings_dict = dict(zip(keys, embeddings))
        return embeddings_dict


class TFHubEncoder(BaseEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hub_url = None
        self.batch_size = kwargs.get('batch_size', DEFAULT_BATCH_SIZE)

    def load_model(self) -> None:
        pdb.set_trace()
        assert self.hub_url is not None, "TF-Hub URL not set"
        # Create graph and finalize (finalizing optional but recommended).
        g = tf.Graph()
        with g.as_default():
            # We will be feeding 1D tensors of text into the graph.
            self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
            self.model = hub.Module(self.hub_url)
            self.embedded_text = self.model(self.text_input)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()
        # Create session and initialize.
        self.session = tf.Session(graph=g)
        self.session.run(init_op)

    def encode(self, keys: List[str], texts: List[str]) -> Union[
        Dict[str, List], Dict[str, Dict[str, List]]]:
        pdb.set_trace()

        use_embeddings_list = list()
        for i, text_batch in enumerate(chunk_gen(texts, self.batch_size)):
            embeddings_batch = self.session.run(self.embedded_text, feed_dict={self.text_input: list(text_batch)})
            use_embeddings_list.extend(embeddings_batch.tolist())
        assert len(use_embeddings_list) == len(texts), "Length mismatch between embeddings and input text"
        use_embeddings_dict = dict(zip(keys, use_embeddings_list))
        return use_embeddings_dict

    def __del__(self):
        self.session.close()


class USEDANEncoder(TFHubEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hub_url = "https://tfhub.dev/google/universal-sentence-encoder/2"


class USELargeEncoder(TFHubEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hub_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"


class ElmoEncoder(BaseEncoder):
    def load_model(self) -> None:
        from allennlp.commands.elmo import ElmoEmbedder
        self.model = ElmoEmbedder()
        self.tokenizer = TreebankWordTokenizer()

    def encode(self, keys: List[str], texts: List[str]) -> Union[
        Dict[str, List], Dict[str, Dict[str, List]]]:
        def elmo_layer_reduce_mean(elmo_vecs):
            all_layers = [ex.mean(axis=0) for ex in elmo_vecs]
            sent_level = [ex.sum(axis=0).tolist() for ex in all_layers]
            return sent_level

        # pdb.set_trace()

        elmo_toks = [self.tokenizer.tokenize(ex) for ex in texts]
        elmo_embeddings = list(self.model.embed_sentences(elmo_toks))

        elmo_embeddings = elmo_layer_reduce_mean(elmo_embeddings)
        elmo_embeddings_dict = dict(zip(keys, elmo_embeddings))
        return elmo_embeddings_dict


class SIFEncoder(BaseEncoder):
    def load_model(self) -> None:
        from sif import data_io, params, SIF_embedding
        self.wordfile = '../data/glove.6B.300d.txt'  # word vector file, can be downloaded from GloVe website
        self.weightfile = '../data/enwiki_vocab_min200.txt'  # each line is a word and its frequency
        self.weightpara = 1e-3  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
        self.rmpc = 1  # number of principal components to remove in SIF weighting scheme
        self.model = SIF_embedding

    def encode(self, keys: List[str], texts: List[str]) -> Union[
        Dict[str, List], Dict[str, Dict[str, List]]]:
        from sif import data_io, params, SIF_embedding
        # load word vectors
        print('loading word vectors')
        (words, We) = data_io.getWordmap(self.wordfile)
        # load word weights
        word2weight = data_io.getWordWeight(self.weightfile,
                                            self.weightpara)  # word2weight['str'] is the weight for the word 'str'
        weight4ind = data_io.getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word
        # load sentences
        x, m = data_io.sentences2idx(texts,
                                     words)  # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
        w = data_io.seq2weight(x, m, weight4ind)  # get word weights

        # set parameters
        print('setting params')
        pars = params.params()
        pars.rmpc = self.rmpc
        # get SIF embedding
        print('getting get SIF embedding')

        sif_embeddings = SIF_embedding.SIF_embedding(We, x, w, pars)  # embedding[i,:] is the embedding for sentence i
        sif_embeddings = [ex.tolist() for ex in sif_embeddings]

        sif_embeddings_dict = dict(zip(keys, list(sif_embeddings)))
        return sif_embeddings_dict
