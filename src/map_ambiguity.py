import copy
import json
import os
import re

import sql_metadata
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from src.config import MappingEnum, DatasetEnum
import src.prompt_tool as prompt_tool
from src.mapping_tool import MappingCorrector
from src.llm import ask_llm
from src.util import load_json, write_to_json, get_best_result, flatten_interpretation
from src.mapping_tool import parse_ambiguity_mapping_str, parse_ambiguity_map, VaguenessDetector


class MapAmbiguity:

    def __init__(self, args):
        self.dev_process_path = args.dev_process_path
        self.column_interpretation_path = args.column_interpretation_path
        self.dev_mapping_path = args.dev_mapping_path
        self.dev_clarification_path = args.dev_clarification_path
        self.db_root_path = args.db_root_path
        self.dataset_name = args.dataset_name
        self.llm_model = args.llm_model
        self.mapping_mode = args.mapping_mode
        self.query_ambiguity = QueryAmbiguity(args)
        self.match_ambiguity = MatchAmbiguity(args)
        self.vagueness_detector = VaguenessDetector()

    def generate_ambiguity_mapping(self):
        print("========start generating ambiguity mapping========")
        dataset = load_json(self.dev_process_path)
        idx = 0
        out = []
        if os.path.exists(self.dev_mapping_path):
            out = load_json(self.dev_mapping_path)
            idx = len(out)
        for i, data in enumerate(tqdm(dataset)):
            if i < idx:
                continue
            question = data["question"]
            query_ambiguity_mapping = "{}"
            data_ambiguity_mapping = "{}"

            if data["ambig_type"] in ["scope", "attachment"]:
                self.mapping_mode = MappingEnum.QUERY
            else:
                self.mapping_mode = MappingEnum.MATCH

            if self.mapping_mode in [MappingEnum.QUERY, MappingEnum.ALL]:
                query_ambiguity_mapping = self.query_ambiguity.get_query_ambiguity_mapping(question)
            if self.mapping_mode in [MappingEnum.MATCH, MappingEnum.ALL]:
                data_ambiguity_mapping = self.match_ambiguity.ambiguity_detection_by_llm(data)

            # TODO merge the query and data ambiguity mapping
            ambiguity_mapping = {"query": query_ambiguity_mapping, "match": data_ambiguity_mapping}

            data["ambiguity_mapping"] = ambiguity_mapping
            try:
                del data["column_interpretation_all"]
                del data["column_interpretation_retrieval"]
            except:
                pass
            out.append(data)
            write_to_json(out, self.dev_mapping_path)

    def correct_mapping(self):
        # use match_ambiguity rules for correction
        self.match_ambiguity.ambiguity_mapping_correction()

    def vagueness_detection(self):
        self.vagueness_detector.de_vague_file(self.dev_mapping_path, self.column_interpretation_path, model=self.llm_model)

    def mapping_revision(self):
        dev_mapping = load_json(self.dev_mapping_path)
        dev_process = load_json(self.dev_process_path)
        dev_clarification = load_json(self.dev_clarification_path)
        out = []
        for data1, data2 in tqdm(zip(dev_mapping, dev_clarification), total=len(dev_mapping)):
            if data1["ambig_type"] in ["scope", "attachment"]:
                out.append(data1)
                continue
            msg = data2["msg"]
            feedback_list = []
            for token in msg:
                if msg[token]["type"] == 2:
                    content = parse_ambiguity_map(msg[token]["content"])
                    table_dict_list = []
                    for table_dict in content[token]:
                        table_dict_list.append(str(table_dict))
                    feedback = f"There are no ambiguities regarding the \"{token}\" mentioned in the question when considering {' and '.join(table_dict_list)}."
                    feedback_list.append(feedback)
            if feedback_list:
                data1["ambiguity_mapping_before_revision"] = copy.deepcopy(data1["ambiguity_mapping"])
                data1["note"] = " ".join(feedback_list)
                data1["column_interpretation_all"] = dev_process[data1["index"]]["column_interpretation_all"]
                data_ambiguity_mapping = self.match_ambiguity.ambiguity_detection_by_llm(data1)
                data1["ambiguity_mapping"]["match"] = data_ambiguity_mapping
                del data1["column_interpretation_all"]
                del data1["column_interpretation_retrieval"]
            out.append(data1)
            write_to_json(out, os.path.join(os.path.dirname(self.dev_mapping_path), "result_mapping_revision.json"))


class QueryAmbiguity:
    def __init__(self, args):
        self.llm_model = args.llm_model

    def get_query_ambiguity_mapping(self, question):
        prompt = prompt_tool.get_prompt_query_ambiguity(question)
        suceess = 0
        temperature = 0
        ambiguity_mapping = "{}"
        while suceess == 0:
            res_list = ask_llm(self.llm_model, prompt, temperature=temperature)
            res = get_best_result(res_list)
            try:
                ambiguity_mapping = parse_ambiguity_mapping_str(res)
                suceess = 1
            except Exception as e:
                temperature = 0.7
        return ambiguity_mapping

class MatchAmbiguity:

    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.llm_model = args.llm_model
        self.retriever = SentenceTransformerRetriever()
        self.column_interpretation_path = args.column_interpretation_path
        self.db_primary_keys_path = args.db_primary_keys_path
        self.dev_process_path = args.dev_process_path
        self.dev_mapping_path = args.dev_mapping_path

    def schema_retrieval(self, question, column_interpretation_all):
        num = int(len(column_interpretation_all) * 0.25)
        if num < 10:
            num = 10
        if num > 15:
            num = 15
        return self.retriever.retrieve_top_k(question, column_interpretation_all, num)

    def schema_retrieval_batch(self, dev_process_path):
        dataset = load_json(dev_process_path)
        for data in tqdm(dataset):
            if data.__contains__("column_interpretation_retrieval"):
                continue
            question = data["question"]
            passages = flatten_interpretation(data["column_interpretation_all"])
            data["column_interpretation_retrieval"] = self.retriever.retrieve_top_k(question, passages, 10)
        write_to_json(dataset, dev_process_path)

    def eval_retrieval(self):
        dataset = load_json(self.dev_process_path)
        retrieval_eval = {"schema_recall": 0, "ambiguity_recall": 0}
        for data in tqdm(dataset):
            question = data["question"]
            passages = flatten_interpretation(data["column_interpretation_all"])
            column_interpretation = self.retriever.retrieve_top_k(question, passages, 10)
            retrieved_schema = []
            for interpretation in column_interpretation:
                try:
                    tmp = re.findall("(.*?): In table \"(.*?)\".", interpretation)
                    table_name = tmp[0][1].strip()
                    column_name = tmp[0][0].strip()
                    retrieved_schema.append(f"{table_name}.{column_name}")
                except:
                    pass
            gold_sqls = get_gold_sqls(self.dataset_name, data)
            gold_schema = get_gold_options_from_sqls(gold_sqls)
            common_schema = [item for item in retrieved_schema if item in gold_schema]
            gold_ambiguity = self.get_gold_ambiguity(gold_sqls)
            common_ambiguity = [item for item in retrieved_schema if item in gold_ambiguity]
            retrieval_eval["schema_recall"] += len(common_schema) / len(gold_schema) / len(dataset)
            retrieval_eval["ambiguity_recall"] += len(common_ambiguity) / len(gold_ambiguity) / len(dataset)
        print(retrieval_eval)

    # def eval_mapping(self):
    #     result = load_json(self.dev_mapping_path)
    #     mapping_eval = {"perfect": 0, "incomplete": 0, "redundant": 0, "wrong": 0}
    #     for data in result:
    #         ambiguity_map_str = data["ambiguity_mapping"]["match"]
    #         ambiguity_map = parse_ambiguity_map(ambiguity_map_str)
    #         try:
    #             ambiguity_list = flatten_map_columns(ambiguity_map)
    #         except:
    #             ambiguity_list = []
    #         flag1 = 1
    #         extra_column_list = get_gold_ambiguity(data["ambiqt_type"], data)
    #         for column in ambiguity_list:
    #             if column not in extra_column_list:
    #                 flag1 = 0
    #         flag2 = 1
    #         for column in extra_column_list:
    #             if column not in ambiguity_list:
    #                 flag2 = 0
    #         if flag1 == 1 and flag2 == 1:
    #             mapping_eval["perfect"] += 1
    #         elif flag1 == 1 and flag2 == 0:
    #             mapping_eval["incomplete"] += 1
    #         elif flag1 == 0 and flag2 == 1:
    #             mapping_eval["redundant"] += 1
    #         elif flag1 == 0 and flag2 == 0:
    #             mapping_eval["wrong"] += 1
    #     for key in mapping_eval:
    #         mapping_eval[key] = round(mapping_eval[key] / len(result) * 100, 2)
    #     print(mapping_eval)

    @staticmethod
    def get_gold_ambiguity(sql_list):
        assert len(sql_list) > 1
        schema_count_map = {}
        for sql in sql_list:
            schemas = get_gold_options_from_sqls([sql])
            for schema in schemas:
                if schema in schema_count_map:
                    schema_count_map[schema] += 1
                else:
                    schema_count_map[schema] = 1
        gold_ambiguity = []
        for schema in schema_count_map:
            if schema_count_map[schema] < len(sql_list):
                gold_ambiguity.append(schema)
        return gold_ambiguity

    def ambiguity_detection_by_llm(self, data):
        question = data["question"]
        passages = flatten_interpretation(data["column_interpretation_all"])
        column_interpretations = self.schema_retrieval(question, passages)
        data["column_interpretation_retrieval"] = column_interpretations
        ambiqt_type = None
        if self.dataset_name == DatasetEnum.AMBIQT and "ambiqt_type" in data:
            ambiqt_type = data["ambiqt_type"]
        if ambiqt_type:
            get_prompt = getattr(prompt_tool, "get_prompt_ambiguity_map_" + ambiqt_type)
        else:
            get_prompt = getattr(prompt_tool, "get_prompt_ambiguity_map")
        if "note" in data:
            note = data["note"]
        else:
            note = None
        if self.dataset_name == DatasetEnum.BIRD:
            note = data["evidence"] if data["evidence"] else "NULL"
            prompt = get_prompt(question, column_interpretations, note)
        else:
            prompt = get_prompt(question, column_interpretations, note)
        suceess = 0
        temperature = 0
        ambiguity_mapping = "{}"
        while suceess == 0:
            res_list = ask_llm(self.llm_model, prompt, temperature=temperature)
            res = get_best_result(res_list)
            try:
                ambiguity_mapping = parse_ambiguity_mapping_str(res)
                suceess = 1
            except Exception as e:
                temperature = 0.7
        return ambiguity_mapping

    def ambiguity_mapping_correction(self):
        # TODO single or batch
        MappingCorrector().correct_mapping_file(self.dev_mapping_path, interpretation_cache_path=self.column_interpretation_path, db_primary_keys_path=self.db_primary_keys_path)

class SentenceTransformerRetriever:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        self.model = SentenceTransformer('E:/ubuntu/实验/result/code_zm/models/all-mpnet-base-v2', device=device)

    def retrieve_top_k(self, question, passages, k):
        embeddings1 = self.model.encode([question])
        embeddings2 = self.model.encode(passages)
        similarity_scores = util.cos_sim(embeddings1, embeddings2)[0].numpy().tolist()
        top_k_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:k]
        top_k_records = [passages[i] for i in top_k_indices]
        return top_k_records

    def retrieve_top_k_with_scores(self, question, passages, k):
        embeddings1 = self.model.encode([question])
        embeddings2 = self.model.encode(passages)
        similarity_scores = util.cos_sim(embeddings1, embeddings2)[0].numpy().tolist()
        top_k_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:k]
        top_k_records = [passages[i] for i in top_k_indices]
        top_k_scores = [similarity_scores[i] for i in top_k_indices]
        return top_k_records, top_k_scores

    def get_similarity_score(self, s1, s2):
        embeddings1 = self.model.encode(s1)
        embeddings2 = self.model.encode(s2)
        similarity_score = util.cos_sim(embeddings1, embeddings2)[0].numpy().tolist()[0]
        return similarity_score

def get_gold_sqls(dataset_name, data):
    if dataset_name == DatasetEnum.AMBIQT:
        gold_sqls = [data["query1"], data["query2"]]
    elif dataset_name == DatasetEnum.AMBROSIA:
        gold_sqls = data["gold_queries"]
    elif dataset_name == DatasetEnum.BIRD:
        gold_sqls = [data["query"]]
    elif dataset_name == DatasetEnum.SPIDER:
        gold_sqls = [data["query"]]
    else:
        raise "Not support " + dataset_name + " now!"
    return gold_sqls


def get_gold_options_from_sqls(gold_sql_list):
    gold_options = []
    for gold_sql in gold_sql_list:
        sql_parsed = sql_metadata.Parser(gold_sql)
        tables = sql_parsed.tables
        columns = sql_parsed.columns
        for column in columns:
            if f"\"{column}\"" in gold_sql:  # value
                continue
            if "." not in column:
                gold_options += [f"{table}.{column}" for table in tables]
            else:
                gold_options.append(column)
    gold_options = list(set(gold_options))
    return gold_options