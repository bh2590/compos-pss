import argparse
import constants

parser = argparse.ArgumentParser()


def encoder_config(local_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    local_parser.add_argument('--encoder_name', type=str,
                              default='use_dan',
                              choices=constants.encoders_dict.keys(),
                              help='encoder type')

    local_parser.add_argument('--dataset_name', type=str,
                              default='sst', choices=['sst', 'phrases'],
                              help='dataset type')

    local_parser.add_argument('--key_prefix', type=str,
                              default='dev',
                              help='key prefix if explicit key not provided')

    # local_parser.add_argument('--keyval_data',
    #                           action='store_true',
    #                           help='if the data-stream is a List of key-text Tuples or just List of texts')

    local_parser.add_argument('--output_embeddings_file', type=str,
                              default='output_embeddings.json',
                              help='output json filename')

    local_parser.add_argument('--batch_size', type=int,
                              default=32,
                              help='batch size for models')

    local_parser.add_argument('--debug',
                              action='store_true',
                              help='whether to enable breakpoints')
    return local_parser


def tre_config(local_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    local_parser.add_argument('--dataset_name', type=str,
                              default='sst', choices=['phrases', 'sst'],
                              help='dataset type')

    local_parser.add_argument('--encoder_name', type=str,
                              default='use_dan',
                              choices=['bert', 'use_dan', 'use_large', 'elmo',
                                       'sif', 'infersent'],
                              help='encoder type')

    local_parser.add_argument('--composition_fn', type=str,
                              default='add', choices=['add', 'mean'],
                              help='composition function')

    local_parser.add_argument('--compositionality_type', type=str,
                              default='node_switching',
                              choices=['node_switching'],
                              help='compositionality calculation type')

    local_parser.add_argument('--output_embeddings_file', type=str,
                              default='output_embeddings.json',
                              help='output json filename')

    local_parser.add_argument('--key_prefix', type=str,
                              default='dev',
                              help='key prefix if explicit key not provided')

    local_parser.add_argument('--debug',
                              action='store_true',
                              help='whether to enable breakpoints')
    return local_parser


def tre_score_gen_config(local_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    local_parser.add_argument('--dataset_name', type=str,
                              default='sst', choices=['sst'],
                              help='dataset type')

    local_parser.add_argument('--compositionality_type', type=str,
                              default='node_switching',
                              choices=['tree_impurity', 'node_switching'],
                              help='compositionality calculation type')

    local_parser.add_argument('--comp_score_file', type=str,
                              default='comp_scores1.json',
                              help='comp_scores json filename')

    local_parser.add_argument('--key_prefix', type=str,
                              default='dev',
                              help='key prefix if explicit key not provided')

    # local_parser.add_argument('--keyval_data',
    #                           action='store_true',
    #                           help='if the data-stream is a List of key-text Tuples or just List of texts')

    local_parser.add_argument('--debug',
                              action='store_true',
                              help='whether to enable breakpoints')

    return local_parser
