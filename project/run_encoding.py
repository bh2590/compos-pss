
import constants
from configs import parser, encoder_config
import ipdb as pdb
from ipdb import slaunch_ipdb_on_exception
from typing import Dict
import json


def main(kwargs: Dict):
    pdb.set_trace()

    dataset = constants.dataset_dict[kwargs['dataset_name']](**kwargs)
    encoder = constants.encoders_dict[kwargs['encoder_name']](**kwargs)

    dataset.load_data()
    encoder.load_model()

    embedding_dict = encoder(dataset)
    prefix = '_'.join([kwargs['dataset_name'], kwargs['encoder_name'], ''])
    output = {}
    # output['metadata'] = {
    #     'prim_len': dataset.prim_len,
    #     'comp_len': dataset.comp_len
    # }
    output['embedding_dict'] = embedding_dict
    with open(prefix + kwargs['output_embeddings_file'], 'w') as f:
        json.dump(dict(output), f, indent=4)


if __name__ == "__main__":
    parser = encoder_config(parser)
    args = parser.parse_args()

    if not args.debug:
        pdb.set_trace = lambda: None

    with slaunch_ipdb_on_exception():
        main(vars(args))