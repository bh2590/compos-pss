from datasets import SSTDataset, PhraseDataset
from encoders import BertServiceEncoder, InferSentEncoder, USEDANEncoder, USELargeEncoder, ElmoEncoder, SIFEncoder
from compositions import Add, Mean

dataset_dict = {
    'sst': SSTDataset,
    'phrases': PhraseDataset,
}

encoders_dict = {
    'bert': BertServiceEncoder,
    'infersent': InferSentEncoder,
    'use_dan': USEDANEncoder,
    'use_large': USELargeEncoder,
    'sif': SIFEncoder,
    'elmo': ElmoEncoder

}

composition_fn_dict = {
    'add': Add,
    'mean': Mean,
}

UNK_TOKEN = '$UNK'
