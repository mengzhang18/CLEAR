import argparse
import os.path

from src.preprocess import PreProcessor
from src.map_ambiguity import MapAmbiguity
from src.reduce_ambiguity import InputWriter, HumanSimulator, VaguenessDetector
from src.config import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=DatasetEnum.CLAMBSQL, help='dataset name')
    parser.add_argument('--db_root_path', type=str, default="./dataset/database/", help='path to databases in dataset')
    parser.add_argument('--dev_path', type=str, default="./dataset/clambsql.json", help='path to dataset input')
    parser.add_argument('--work_dir_path', type=str, default="./data/clambsql/", help='path to column meaning')
    parser.add_argument('--column_interpretation_path', type=str, help='path to column meaning')
    parser.add_argument('--db_primary_keys_path', type=str, help='path to primary keys')
    parser.add_argument('--dev_process_path', type=str, help='path to databases preprocess')
    parser.add_argument('--dev_mapping_path', type=str, help='path to ambiguity mapping')
    parser.add_argument('--dev_clarification_path', type=str, help='path to ambiguity mapping')
    parser.add_argument('--dev_rewriting_path', type=str, help='path to disambiguation output')
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini', help='openai model')
    parser.add_argument('--mapping_mode', type=str, default=MappingEnum.ALL)
    parser.add_argument('--schema_with_content', action='store_true', default=True)
    args = parser.parse_args()
    return args

def run(args):
    PreProcessor(args).preprocess()
    MapAmbiguity(args).generate_ambiguity_mapping()
    MapAmbiguity(args).correct_mapping()
    MapAmbiguity(args).vagueness_detection()
    InputWriter(args).rewrite_clear()
    HumanSimulator(args).election_clambsql()
    MapAmbiguity(args).mapping_revision()


if __name__ == "__main__":
    args = parse_args()

    if args.work_dir_path:
        args.column_interpretation_path = os.path.join(args.work_dir_path, "interpretation_cache.json")
        args.dev_process_path = os.path.join(args.work_dir_path, "validation_process.json")
        args.dev_mapping_path = os.path.join(args.work_dir_path, "result_mapping.json")
        args.dev_rewriting_path = os.path.join(args.work_dir_path, "result_rewriting.json")
        args.dev_clarification_path = os.path.join(args.work_dir_path, "result_clarification.json")
        args.db_primary_keys_path = os.path.join(args.work_dir_path, "db_primary_keys.json")

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    run(args)
