import copy
import json
import re
import sqlite3

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.util import get_best_result
from src.llm import ask_llm
from src.stop_words import *
from src.util import load_json, write_to_json, schema_without_content_to_map, tokenize_question

def parse_ambiguity_mapping_str(dq_str:str):
    dq_str = dq_str.replace("\n", " ")
    ambiguity_mapping_str = re.findall("```json(.*?)```", dq_str)[0]
    ambiguity_mapping_str = " ".join(ambiguity_mapping_str.split()).strip()
    ambiguity_mapping = json.loads(ambiguity_mapping_str)
    if "Ambiguity Mapping" in ambiguity_mapping:
        ambiguity_mapping_str_new = json.dumps(ambiguity_mapping["Ambiguity Mapping"])
    else:
        ambiguity_mapping_str_new = json.dumps(ambiguity_mapping)
    return ambiguity_mapping_str_new

def parse_ambiguity_map(ambiguity_map_str):
    if isinstance(ambiguity_map_str, str):
        return json.loads(ambiguity_map_str)
    elif isinstance(ambiguity_map_str, dict):
        ambiguity_map = {}
        ambiguity_map["query"] = json.loads(ambiguity_map_str["query"])
        ambiguity_map["match"] = json.loads(ambiguity_map_str["match"])
        return ambiguity_map

def filter_ambiguity_map(ambiguity_map_str):
    ambiguity_map = parse_ambiguity_map(ambiguity_map_str)
    ambiguity_map_new = {}
    for token in ambiguity_map:
        token_ambiguity = []
        if not isinstance(ambiguity_map[token], list):
            continue
        for i, table_dict in enumerate(ambiguity_map[token]):
            if not isinstance(table_dict, dict):
                continue
            table_dict_new = {}
            for table in table_dict:
                columns = table_dict[table]
                if not isinstance(columns, list):
                    if isinstance(columns, str):
                        columns = [columns]
                    else:
                        continue
                columns = list(set(columns))
                if columns:
                    table_dict_new[table] = columns
            if table_dict_new and table_dict_new not in token_ambiguity:
                token_ambiguity.append(table_dict_new)
        if len(token_ambiguity) >= 2:
            ambiguity_map_new[token] = token_ambiguity
    return json.dumps(ambiguity_map_new)


def extend_primary_keys(primary_keys, schema_without_content):
    schema_map = schema_without_content_to_map(schema_without_content)
    for table in schema_map:
        for column in schema_map[table]:
            if table not in primary_keys:
                primary_keys[table] = []
            if ((column.lower().endswith("_id") or column.endswith("Id") or column.lower() == "id")
                    and column not in primary_keys[table]):
                primary_keys[table].append(column)
    return primary_keys

def get_primary_keys_required(primary_keys, question):
    primary_keys_required = {}
    for table in primary_keys:
        table_primary_keys_required = []
        table_primary_keys = primary_keys[table]
        for primary_key in table_primary_keys:
            primary_key_tmp = primary_key.replace("_", " ")
            if primary_key_tmp.lower() in question.lower():
                table_primary_keys_required.append(primary_key)
        if table_primary_keys_required:
            primary_keys_required[table] = table_primary_keys_required
    return primary_keys_required

def convert_ambiguity_map_old_to_new(ambiguity_map_str):
    ambiguity_map = parse_ambiguity_map(ambiguity_map_str)
    ambiguity = {}
    for token in ambiguity_map:
        ambiguity[token] = []
        for table in ambiguity_map[token]:
            column = ambiguity_map[token][table]
            if type(column) == str:
                ambiguity[token].append({table: [column]})
            elif type(column) == list and table != "_AND_":
                for v in column:
                    ambiguity[token].append({table: [v]})
            elif table == "_AND_":
                if isinstance(column, list):
                    for table_dict in column:
                        ambiguity[token].append(table_dict)
                else:
                    # a = {}
                    # for tbl in column:
                    #     a[tbl] = []
                    #     col = column[tbl]
                    #     if type(col) == str:
                    #         a[tbl].append(col)
                    #     elif type(col) == list:
                    #         for v in column:
                    #             a[tbl].append(v)
                    ambiguity[token].append(column)
    return json.dumps(ambiguity)

def get_query_ambiguity_type(query_ambiguity_map):
    for token in query_ambiguity_map:
        for p in query_ambiguity_map[token]:
            if p.startswith("for each") or p.startswith("common to all"):
                return "scope"
            else:
                for p2 in query_ambiguity_map[token]:
                    if "and" in p2.split() or "or" in p2.split():
                        return "attachment"
    return None

def convert_attach_ambiguity_old_to_new(question, query_ambiguity_map_str):
    query_ambiguity_map = parse_ambiguity_map(query_ambiguity_map_str)
    if get_query_ambiguity_type(query_ambiguity_map) == "attachment":
        query_ambiguity_map_new = {}
        token = [*query_ambiguity_map][0]
        interpretations = query_ambiguity_map[token]
        attachment_words = ["and", "or"]
        for attachment_word in attachment_words:
            for p in interpretations:
                if attachment_word in p.split():
                    e1 = p.split(attachment_word)[0].strip()
                    e2 = p.split(attachment_word)[1].strip()
                    if f"{p} {token}" in question:
                        e_token = f"{p} {token}"
                        query_ambiguity_map_new[e_token] = [f"{e2} {token} {attachment_word} {e1} {token}",
                                                            f"{e2} {token} {attachment_word} {e1}"]
                    elif f"{token} {p}" in question:
                        e_token = f"{token} {p}"
                        query_ambiguity_map_new[e_token] = [f"{token} {e1} {attachment_word} {token} {e2}",
                                                            f"{e2} {attachment_word} {e2} {token}"]
        query_ambiguity_map_new_str = json.dumps(query_ambiguity_map_new)
    else:
        query_ambiguity_map_new_str = query_ambiguity_map_str
    return query_ambiguity_map_new_str

class MappingCorrector:

    def __init__(self):
        from stanfordcorenlp import StanfordCoreNLP

        self.retriever = SentenceTransformer()
        self.nlp = StanfordCoreNLP('E:/ubuntu/ambiguity/stanford-corenlp-4.5.6')

    def correct_mapping_file(self, result_path, mode="all", interpretation_cache_path=None, db_primary_keys_path=None):
        result = load_json(result_path)
        interpretation_cache = None
        db_primary_keys = None
        if interpretation_cache_path:
            print("Use column enrichment information")
            interpretation_cache = load_json(interpretation_cache_path)
        if db_primary_keys_path:
            print("Use primary key information")
            db_primary_keys = load_json(db_primary_keys_path)
        for data in tqdm(result):
            if mode in ["query", "all"]:
                ambiguity_mapping_q = self.correct_query_mapping_data(data["question"], data["ambiguity_mapping"]["query"])
                data["ambiguity_mapping"]["query"] = ambiguity_mapping_q
            elif mode in ["match", "all"]:
                ambiguity_mapping_m = self.correct_match_mapping_data(data, interpretation_cache, db_primary_keys)
                data["ambiguity_mapping"]["match"] = ambiguity_mapping_m
        write_to_json(result, result_path)

    def correct_query_mapping_data(self, question, query_ambiguity_map_str):
        query_ambiguity_map = parse_ambiguity_map(query_ambiguity_map_str)
        if not query_ambiguity_map:
            return "{}"
        token = [*query_ambiguity_map][0]
        if token not in question:
            return "{}"
        interpretations = query_ambiguity_map[token]
        if not isinstance(interpretations, list) or len(interpretations) < 2:
            return "{}"
        ambig_type = get_query_ambiguity_type({token: interpretations})
        interpretations_new = []
        if ambig_type == "scope":
            scope_words = ["each", "every", "all"]
            for scope_word in scope_words:
                if scope_word in token.split():
                    token = token[token.find(scope_word):].strip()
                    entity = token[token.find(scope_word) + len(scope_word):].strip()
                    for p in interpretations:
                        if p.startswith(f"for {scope_word}") or p.startswith(f"for each") :
                            p = f"for each {entity} individually"
                        elif p.startswith(f"common to all"):
                            p = ' '.join(f"common to all {' '.join(entity.split()[:-1])} {p.split()[-1]}".split())
                        interpretations_new.append(p)
        elif ambig_type == "attachment":
            attachment_words = ["and", "or"]
            for attachment_word in attachment_words:
                if attachment_word in token.split():
                    interpretations_new = interpretations
                    break
                else:
                    query_ambiguity_map_new_str = convert_attach_ambiguity_old_to_new(question, query_ambiguity_map_str)
                    return query_ambiguity_map_new_str
        else:
            return "{}"
        return json.dumps({token: interpretations_new})

    def correct_match_mapping_data(self, data, interpretation_cache=None, db_primary_keys=None):
        db_id = data["db_id"]
        question = data["question"]
        schema_without_content = data["schema_without_content"]
        schema_map = schema_without_content_to_map(schema_without_content)
        question_tokens = tokenize_question(self.nlp, question, ["-"])
        ambiguity_map_str = data["ambiguity_mapping"]["match"]
        ambiguity_map_str = filter_ambiguity_map(ambiguity_map_str)
        ambiguity_map = parse_ambiguity_map(ambiguity_map_str)
        ambiguity_map_new = {}
        for token in ambiguity_map:
            token_ambiguity = []
            # check the table and column existence
            for table_dict in ambiguity_map[token]:
                table_dict_new = {}
                for table in table_dict:
                    table_dict_new[table] = []
                    columns = table_dict[table]
                    for column in columns:
                        if table in schema_map and column in schema_map[table]:
                            table_dict_new[table].append(column)
                    if not table_dict_new[table]:
                        del table_dict_new[table]
                if table_dict_new:
                    token_ambiguity.append(table_dict_new)
            if len(token_ambiguity) < 2:
                continue
            # match the token with question
            question_tokens_filter = []
            for tok in question_tokens:
                if (tok.lower() not in get_symbols()
                        and tok.lower() not in get_stop_words()
                        and tok not in ambiguity_map_new):
                    question_tokens_filter.append(tok)
            token = " ".join(token.split("_"))
            if " " in token and token in question:
                token_new = token
            elif token not in question_tokens_filter:
                tbl = list(token_ambiguity[0].keys())[0]
                col = token_ambiguity[0][tbl][0]
                token_scores = self.retriever.retrieve_top_k_with_scores(token, question_tokens_filter, 1)
                if interpretation_cache:
                    col_interpretation = interpretation_cache[db_id][tbl][col]
                    interpretation_scores = self.retriever.retrieve_top_k_with_scores(col_interpretation, question_tokens_filter, 1)
                    if interpretation_scores[1][0] > token_scores[1][0]:
                        token_new = interpretation_scores[0][0]
                    else:
                        token_new = token_scores[0][0]
                else:
                    token_new = token_scores[0][0]
            else:
                token_new = token
            ambiguity_map_new[token_new] = token_ambiguity
        ambiguity_map_str = filter_ambiguity_map(json.dumps(ambiguity_map_new))
        if db_primary_keys:
            primary_keys = db_primary_keys[db_id]
            primary_keys = extend_primary_keys(primary_keys, schema_without_content)
            ambiguity_map_str = self.correct_primary_key(ambiguity_map_str, question, primary_keys)
        return ambiguity_map_str

    def correct_primary_key(self, ambiguity_map_str, question, primary_keys):
        match_ambiguity_map = parse_ambiguity_map(ambiguity_map_str)
        primary_keys_required = get_primary_keys_required(primary_keys, question)
        match_ambiguity_map_copy = copy.deepcopy(match_ambiguity_map)
        for token in match_ambiguity_map:
            for i, table_dict in enumerate(match_ambiguity_map[token]):
                for table in table_dict:
                    for column in table_dict[table]:
                        if column in primary_keys[table]:
                            if table not in primary_keys_required or column not in primary_keys_required[table]:
                                match_ambiguity_map_copy[token][i][table].remove(column)
        match_ambiguity_str = json.dumps(match_ambiguity_map_copy)
        ambiguity_map_str_new = filter_ambiguity_map(match_ambiguity_str)
        return ambiguity_map_str_new

        # # deal table id
        # ambiguity_map = parse_ambiguity_map(ambiguity_map_str)
        # ambiguous_columns = flatten_map_columns(ambiguity_map)
        # flag = 0
        # for column in ambiguous_columns:
        #     if column.endswith("id"):
        #         flag = 1
        # if flag == 1:
        #     ambiguity_map_new = {}
        #     for token in ambiguity_map:
        #         ambiguity_map_new[token] = {}
        #         for table in ambiguity_map[token]:
        #             v = ambiguity_map[token][table]
        #             if type(v) == list:
        #                 v_new = []
        #                 for col in v:
        #                     if not col.endswith("id"):
        #                         v_new.append(col)
        #                 ambiguity_map_new[token][table] = v_new
        #             else:
        #                 if not v.endswith("id"):
        #                     ambiguity_map_new[token][table] = v
        #     ambiguity_map_str_new = filter_ambiguity_map(json.dumps(ambiguity_map_new))




class VaguenessDetector:
    def __init__(self, args):
        self.retriever = SentenceTransformer()

    def de_vague_file(self, result_path, interpretation_path, model="gpt-4o"):
        result = load_json(result_path)
        interpretation_db = load_json(interpretation_path)
        for data in tqdm(result):
            question = data["question"]
            interpretations = interpretation_db[data["db_id"]]
            ambiguity_map_str = data["ambiguity_mapping"]["match"]
            data_ambiguity_map_str = self.de_vague_data(ambiguity_map_str, question, interpretations, model)
            data["ambiguity_mapping"]["match"] = data_ambiguity_map_str
            write_to_json(result, result_path)
        write_to_json(result, result_path)

    def de_vague_data_llm(self, ambiguity_map_str, question, interpretations, model="gpt-4o"):
        data_ambiguity_map = parse_ambiguity_map(ambiguity_map_str)
        options_map = get_options_map(data_ambiguity_map)
        data_ambiguity_map_new = {}
        for token in options_map:
            prompt = self.get_prompt_vague(question, token, options_map[token], interpretations)
            res_list = ask_llm(model, prompt, n=10, temperature=0.7)
            res = get_best_result(res_list)
            schema_tuple = self.get_mapping_by_options_res(res, options_map[token])
            data_ambiguity_map_new[token] = schema_tuple
        data_ambiguity_map_str = json.dumps(data_ambiguity_map_new)
        data_ambiguity_map_str = filter_ambiguity_map(data_ambiguity_map_str)
        return data_ambiguity_map_str

    def is_vague_by_content(self, ambiguity_map_str, db_path):
        data_ambiguity_map = parse_ambiguity_map(ambiguity_map_str)
        result = {}  # 0:no 1:yes -1:don't know
        for token in data_ambiguity_map:
            # can not know
            result[token] = -1
            # the number of columns in table_dict should be equal
            count = []
            for table_dict in data_ambiguity_map[token]:
                column_list = convert_table_dict_to_list(table_dict)
                if len(column_list) not in count:
                    count.append(len(column_list))
            if len(count) != 1:
                result[token] = 1  # vague = True
                continue
            # the content of columns should be different
            value_similarity = 0
            for i in range(len(data_ambiguity_map[token])):
                if i > len(data_ambiguity_map[token]) - 2:
                    continue
                table_dict_1 = data_ambiguity_map[token][i]
                table_dict_2 = data_ambiguity_map[token][i+1]
                table_dict_values_1 = get_table_dict_values(table_dict_1, db_path)
                table_dict_values_2 = get_table_dict_values(table_dict_2, db_path)
                if not table_dict_values_1[0] or not table_dict_values_2[0]:
                    break
                if table_dict_values_1 == table_dict_values_2:
                    value_similarity = 1
                else:
                    s1 = table_dict_values_1.__str__()
                    s2 = table_dict_values_2.__str__()
                    value_similarity = self.retriever.get_similarity_score(s1, s2)
            if value_similarity >= 0.9:
                result[token] = 0
        return result

    @staticmethod
    def get_vague_table_dict(schema_tuple):
        table_dict_new = {}
        for table_dict in schema_tuple:
            for table in table_dict:
                if table not in table_dict_new:
                    table_dict_new[table] = table_dict[table]
                else:
                    for column in table_dict[table]:
                        if column not in table_dict_new[table]:
                            table_dict_new[table].append(column)
        return table_dict_new

    @staticmethod
    def get_prompt_vague(question, token, options, interpretations):
        options_str = ""
        for i, option_tuple in enumerate(options):
            interpretation_tuple = []
            for option in option_tuple:
                table = option.split(".")[0]
                column = option.split(".")[1]
                interpretation = interpretations[table][column]
                interpretation_tuple.append(interpretation)
            interpretation_tuple_str = "\n\t".join(interpretation_tuple)
            options_str += f"[{i + 1}] {interpretation_tuple_str}\n"
        prompt = ('''### Ambiguity Resolution Task
    When you are performing a text-to-SQL task, you encounter ambiguity during schema linking.
    For the provided ambiguous question and keyword, select the appropriate option to clarify the keyword.
    Each option consists of the option number, the column name, and the column meaning.
    Each option can contain multiple columns, separated by a newline.
    Follow these rules:
    1. Single choice and multiple choice are both allowed.
    2. Respond with the option number.
    3. If some options are all suitable for the keyword, respond with "and", such as [1] and [2] and ...
    4. If some options are either suitable for the keyword, respond with "or", such as [1] or [2] or ...
    5. You can also respond with the combination of "and" and "or", such as ([2] or [3]) and [1] or ...

    Question: [QUESTION]

    Keyword: [TOKEN]

    Options:
    [OPTIONS]

    Answer: '''
                  .replace("[QUESTION]", question)
                  .replace("[TOKEN]", token)
                  .replace("[OPTIONS]", options_str))
        return prompt

    @staticmethod
    def get_mapping_by_options_res(res, options):
        if "(" in res:
            # TODO
            idx_str_list = re.findall("\[(.*?)\]", res)
            idx_list = [int(idx_str) - 1 for idx_str in idx_str_list]
            options_new = [options[idx] for idx in idx_list]
        else:
            options_new = []
            res_or = res.split("or")
            for sub_res in res_or:
                if not re.search("\[(.*?)\]", sub_res):
                    continue
                if "and" in sub_res:
                    res_and = sub_res.split("and")
                    list_and = []
                    for sub_res_and in res_and:
                        try:
                            idx_str = re.findall("\[(.*?)\]", sub_res_and)[0]
                        except:
                            continue
                        idx = int(idx_str) - 1
                        options_new.append(options[idx])
                        list_and.extend(options[idx])
                    options_new.append(list_and)
                else:
                    idx_str = re.findall("\[(.*?)\]", sub_res)[0]
                    idx = int(idx_str) - 1
                    if idx == len(options):
                        print("# additional option \"[n+1] all of them\"")
                        options_new = options
                        options_new.append([option for option_tuple in options for option in option_tuple])
                    else:
                        options_new.append(options[idx])
        schema_tuple = convert_list_to_schema_tuple(options_new)
        return schema_tuple

def get_table_dict_values(table_dict, db_path):
    table_dict_values = []
    for table in table_dict:
        for column in table_dict[table]:
            values = get_column_values(db_path, table, column)
            table_dict_values.append(values)
    return table_dict_values

def get_column_values(db_path, table, column):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        sql = f"SELECT `{column}` FROM `{table}` WHERE `{column}` is not null ORDER BY `{column}` LIMIT 100"
        cursor.execute(sql)
        rows = cursor.fetchall()
        values = [row[0] for row in rows]
        return values
    except Exception as e:
        print(e)
        return []
    finally:
        conn.close()

def get_options_map(data_ambiguity_map):
    options_map = {}
    for token in data_ambiguity_map:
        schema_tuple = data_ambiguity_map[token]
        options = convert_schema_tuple_to_list(schema_tuple)
        options_map[token] = options
    return options_map

def convert_schema_tuple_to_list(schema_tuple):
    options = []
    for table_dict in schema_tuple:
        option_tuple = []
        for table in table_dict:
            columns = table_dict[table]
            for column in columns:
                option_tuple.append(f"{table}.{column}")
        options.append(option_tuple)
    return options

def convert_list_to_schema_tuple(options):
    schema_tuple = []
    for option_tuple in options:
        table_dict = convert_list_to_table_dict(option_tuple)
        schema_tuple.append(table_dict)
    return schema_tuple

def convert_list_to_table_dict(option_tuple):
    col_map = {}
    for col in option_tuple:
        t = col.split(".")[0]
        c = col.split(".")[1]
        if t not in col_map:
            col_map[t] = []
        col_map[t].append(c)
    return col_map

def convert_table_dict_to_list(table_dict):
    option_tuple = []
    for table in table_dict:
        columns = table_dict[table]
        for column in columns:
            option_tuple.append(f"{table}.{column}")
    return option_tuple



if __name__ == '__main__':
    result_path = "../data/clambsql/clear/mini/result_mapping.json"
    # result = load_json(result_path)
    # for data in result:
    #     if data["ambig_type"] == "attachment":
    #         data["gold_ambiguity"]["query"] = convert_attach_ambiguity_old_to_new(data["question"], data["gold_ambiguity"]["query"])
    #         # data["ambiguity_mapping"]["query"] = convert_attach_ambiguity_old_to_new(data["question"], data["ambiguity_mapping"]["query"])
    # write_to_json(result, result_path)

    # MappingCorrector().correct_mapping_file(result_path, mode="query")
    dataset = load_json("../dataset/clambsql.json")
    corrector = MappingCorrector()
    for data in dataset:
        if data["index"] < 1251:
            continue
        if data["ambig_type"] in ["scope", "attachment"]:
            data["gold_ambiguity"]["query"] = corrector.correct_query_mapping_data(data["question"], data["gold_ambiguity"]["query"])
    write_to_json(dataset, "../dataset/clambsql.json")
#     db_root_path = "E:/ubuntu/dataset/bird/dev/dev_databases/"
#     result_path = "../data/bird/evidence/result_mapping.json"
#
#     # db_root_path = "../../dataset/ambiQT/db-content/database/"
#     # result_path = "../data/ambiqt/column/result_mapping.json"
#     result = load_json(result_path)
#     data = result[28]
#     db_path = db_root_path + data["db_id"] + "/" + data["db_id"] + ".sqlite"
#     match_ambiguity_map_str = data["ambiguity_mapping"]["match"]
#     # match_ambiguity_map_str = convert_ambiguity_map_old_to_new(match_ambiguity_map_str)
#
#     # result_path = "../data/ambrosia/vague/test/result_mapping.json"
#     # interpretation_path = "../data/ambrosia/vague/interpretation.json"
#     vaguer = VaguenessDetector()
#     # corrector.de_vague_file(result_path, interpretation_path)
#     is_vague = vaguer.is_vague_by_content(match_ambiguity_map_str, db_path)
#     print(is_vague)