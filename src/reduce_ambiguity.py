import json
import os
import random
import time

import sql_metadata
from tqdm import tqdm

from src.config import DatasetEnum, SelectionEnum
from src.map_ambiguity import SentenceTransformerRetriever
from src.mapping_tool import parse_ambiguity_map, VaguenessDetector, convert_list_to_schema_tuple,get_options_map
from src.preprocess import get_db_path
from src.llm import ask_llm
from src.util import load_json, write_to_json
from src.rewriter import Rewriter
import src.app.app as app

app.inputs = {}
app.outputs = {}


class ReduceAmbiguity:

    def __init__(self, args):
        self.dev_process_path = args.dev_process_path
        self.dev_mapping_path = args.dev_mapping_path
        self.output_path = args.output_path
        self.db_root_path = args.db_root_path
        self.dataset_name = args.dataset_name
        self.schema_with_content = args.schema_with_content
        self.selection_mode = args.selection_mode
        self.human_selection_mode = args.human_selection_mode
        self.voting_selection_mode = args.voting_selection_mode


    def ambiguity_selection(self):
        print("========start reducing ambiguity========")
        selection_method = getattr(self, self.selection_mode + "_selection")
        selection_method()

    def human_selection(self):
        mapping_results = load_json(self.dev_mapping_path)
        # get inputs
        idx = 0
        if os.path.exists(self.output_path):
            out = load_json(self.output_path)
            idx = len(out)
        ambiguity_inputs = {}
        for i, data in enumerate(mapping_results):
            if i < idx:
                continue
            question = data["question"]
            ambiguity_mapping = data["ambiguity_mapping"]
            ambiguity_map = parse_ambiguity_map(ambiguity_mapping)
            ambiguity_map_new = {}
            for token in ambiguity_map:
                ambiguity_map_new[token] = {}
                options = []
                for table in ambiguity_map[token]:
                    v = ambiguity_map[token][table]
                    if isinstance(v, str):
                        v = [v]
                    for column in v:
                        options.append(f"{table}.{column}")
                amb_id = f"{i}:{token}"
                ambiguity_inputs[amb_id] = [question, token, options]
        # interactive clarification
        if self.human_selection_mode == SelectionEnum.HumanSelector.APP:
            app.inputs = ambiguity_inputs
            app.run_asyn()
            while len(app.outputs) != len(ambiguity_inputs):
                sleep_time = (len(ambiguity_inputs) - len(app.outputs))
                if sleep_time > 30:
                    sleep_time = 30
                time.sleep(sleep_time)
                self.save_human_clarification_data(app.outputs)
            self.save_human_clarification_data(app.outputs)
        elif self.human_selection_mode == SelectionEnum.HumanSelector.INPUT:
            for amb_id in ambiguity_inputs:
                print(f"=====start clarifying question {amb_id}=====")
                question, token, options = ambiguity_inputs[amb_id]
                while 1:
                    option = input(f"Question: {question} | Which '{token}'? {options}:")
                    if option in options or not options:
                        break
                    else:
                        print("invalid option!")
                self.save_human_clarification_data(app.outputs)
        else:
            raise "Not supported " + self.human_selection_mode

    def save_human_clarification_data(self, output_options_map):
        mapping_results = load_json(self.dev_mapping_path)
        idx = 0
        out = []
        if os.path.exists(self.output_path):
            out = load_json(self.output_path)
            idx = len(out)
        for i, data in enumerate(mapping_results):
            if i < idx:
                continue
            ambiguity_mapping = data["ambiguity_mapping"]
            ambiguity_map = parse_ambiguity_map(ambiguity_mapping)
            ambiguity_map_new = {}
            for token in ambiguity_map:
                ambiguity_map_new[token] = {}
                amb_id = f"{i}:{token}"
                if amb_id not in output_options_map:
                    break
                final_option = output_options_map[amb_id]
                final_table = final_option.split(".")[0]
                final_column = final_option.split(".")[1]
                ambiguity_map_new[token][final_table] = final_column
                data["ambiguity_clarification"] = json.dumps(ambiguity_map_new)
                out.append(data)
            write_to_json(out, self.output_path)


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

class HumanSimulator:
    def __init__(self, args):
        self.dev_mapping_path = args.dev_mapping_path
        self.dev_clarification_path = args.dev_clarification_path
        self.db_root_path = args.db_root_path
        self.dataset_name = args.dataset_name

    def election_bird(self):
        result = load_json(self.dev_mapping_path)
        vaguer = VaguenessDetector()
        for data in tqdm(result):
            # print(data["query"])
            gold_sql = data["query"]
            sql_parsed = sql_metadata.Parser(gold_sql)
            tokens = sql_parsed.tokens
            sql_tokens = []
            for token in sql_parsed.non_empty_tokens:
                if token.value.startswith("`"):
                    column = token.value.strip("`")
                    idx = gold_sql.index(column)
                    if gold_sql[idx - 2] != ".":
                        sql_tokens.append(column)
            try:
                columns_parsed = list(sql_parsed.columns)
            except:
                columns_parsed = []
                pass
            tables_parsed = sql_parsed.tables
            columns_all = []
            for column in columns_parsed + sql_tokens:
                if "." not in column:
                    column = f"{tables_parsed[0]}.{column}"
                if column not in columns_all:
                    columns_all.append(column)
            ambiguity_mapping = parse_ambiguity_map(data["ambiguity_mapping"])
            # print(schema_tokens)
            match_ambiguity = ambiguity_mapping["match"]
            options_map = get_options_map(match_ambiguity)
            msg = {}
            ambiguity_clarification = {}
            for token in options_map:
                msg[token] = {"type": -1}
                options = options_map[token]
                flag = 0
                col_list = []
                for option in options:
                    sub_flag = 1
                    for col in option:
                        if col not in columns_all:
                            sub_flag = 0
                    if sub_flag == 1:
                        flag += 1
                        col_list.append(option)
                content = {token: convert_list_to_schema_tuple(col_list)}
                if flag == 0:
                    msg[token]["type"] = 2
                    msg[token]["content"] = {token: match_ambiguity[token]}
                    # ambiguity_clarification[token] = {}
                if flag == 1:
                    msg[token]["type"] = 0
                    msg[token]["content"] = content
                    ambiguity_clarification[token] = content[token][0]
                if flag > 1:
                    msg[token]["type"] = 1
                    msg[token]["content"] = content
                    db_path = get_db_path(DatasetEnum.BIRD, self.db_root_path, data["db_id"])
                    is_vague = vaguer.is_vague_by_content(json.dumps(content), db_path)
                    if is_vague != 0:
                        ambiguity_clarification[token] = vaguer.get_vague_table_dict(content[token])
                    else:
                        ambiguity_clarification[token] = content[token][0]
                msg[token]["content"] = json.dumps(msg[token]["content"])
            data["msg"] = msg
            del data["schema_with_content"]
            data["ambiguity_clarification"] = json.dumps(ambiguity_clarification)
        write_to_json(result, self.dev_clarification_path)

    def election_clambsql(self):
        result = load_json(self.dev_mapping_path)
        retriever = SentenceTransformerRetriever()
        for data in tqdm(result):
            # if data["index"] < 899:
            #     continue
            clear_ambiguity = parse_ambiguity_map(data["clear_ambiguity"])
            ambiguity_mapping = parse_ambiguity_map(data["ambiguity_mapping"])
            query_ambiguity = ambiguity_mapping["query"]
            match_ambiguity = ambiguity_mapping["match"]
            msg = {}
            ambiguity_clarification = {}
            for token1 in query_ambiguity:
                msg[token1] = {}
                interpretations = query_ambiguity[token1]
                flag = 0
                for token2 in clear_ambiguity:
                    p = clear_ambiguity[token2]
                    # if p in interpretations:
                    #     flag = 1
                    for interpretation in interpretations:
                        if retriever.get_similarity_score(p, interpretation) > 0.9:
                            flag = 1
                if flag == 0:
                    msg[token1]["type"] = 2
                    msg[token1]["content"] = {token1: interpretations}
                else:
                    msg[token1]["type"] = 0
                    msg[token1]["content"] = {token1: interpretation}
                    ambiguity_clarification[token1] = interpretation
                msg[token1]["content"] = json.dumps(msg[token1]["content"])
            for token1 in match_ambiguity:
                msg[token1] = {}
                flag = 0
                table_dicts1 = match_ambiguity[token1]
                for token2 in clear_ambiguity:
                    table_dict2 = clear_ambiguity[token2]
                    if table_dict2 in table_dicts1:
                        flag = 1
                        content = table_dict2
                    else:
                        sub_flag = 0
                        for table in table_dict2:
                            for col in table_dict2[table]:
                                if {table: [col]} not in table_dicts1:
                                    sub_flag = 1
                        if sub_flag == 0:
                            flag = 2
                            content = table_dict2
                if flag == 0:
                    msg[token1]["type"] = 2
                    msg[token1]["content"] = {token1: table_dicts1}
                    # ambiguity_clarification[token] = {}
                if flag == 1:
                    msg[token1]["type"] = 0
                    msg[token1]["content"] = {token1: content}
                    ambiguity_clarification[token1] = content
                if flag == 2:
                    msg[token1]["type"] = 1
                    msg[token1]["content"] = {token1: content}
                    ambiguity_clarification[token1] = content
                msg[token1]["content"] = json.dumps(msg[token1]["content"])
            data["msg"] = msg
            data["ambiguity_clarification"] = json.dumps(ambiguity_clarification)
        write_to_json(result, self.dev_clarification_path)


class InputWriter:
    def __init__(self, args):
        self.dev_mapping_path = args.dev_mapping_path
        self.dev_clarification_path = args.dev_clarification_path
        self.dev_rewriting_path = args.dev_rewriting_path
        self.dataset_name = args.dataset_name
        self.schema_with_content = args.schema_with_content

    def rewrite_all(self, mode="all"):
        rewriter = Rewriter(input_path=self.dev_mapping_path, output_path=self.dev_rewriting_path,
                            with_content=self.schema_with_content)
        rewriter.rewrite_ambiguity(mode)

    def rewrite_clear(self, mode="all"):
        rewriter = Rewriter(input_path=self.dev_clarification_path, output_path=self.dev_rewriting_path,
                            with_content=self.schema_with_content)
        rewriter.rewrite_clarification(mode)
