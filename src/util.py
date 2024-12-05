import json
import re
import regex
import sqlparse
from sqlparse.sql import Parenthesis

from tqdm import tqdm

SQL_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'join', 'on',
                'as', 'not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', 'none', '-',
                '+', '*', '/', 'none', 'max', 'min', 'count', 'sum', 'avg', 'and', 'or', 'desc', 'asc')

def get_sql_keywords():
    return SQL_KEYWORDS

def load_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)

def write_to_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.flush()

def insert_after_key(original_dict, key_to_find, new_key, new_value):
    items = list(original_dict.items())
    index_to_insert = next((i for i, (k, _) in enumerate(items) if k == key_to_find), None)
    if index_to_insert is not None:
        items.insert(index_to_insert + 1, (new_key, new_value))
    else:
        items.append((new_key, new_value))
    return dict(items)

def parse_schema_with_content_to_map(schema_with_content):
    schema = []
    tables = schema_with_content.split("|")
    for table in tables:
        table_name = table.split(":")[0].strip()
        column_with_content = table[table.index(":") + 1:]
        columns = column_with_content.split("), ")
        columns[-1] = columns[-1][:columns[-1].rindex(")")]
        schema_columns = []
        for column in columns:
            column = column + ")"
            cells_str = regex.findall(r'\((?:[^()]|(?R))*\)', column)[-1]
            column_name = column.replace(cells_str, "").strip()
            # cells = cells_str.strip("(").strip(")").split(",")
            # cells = [cell.strip().replace("\"", "") for cell in cells]
            schema_columns.append({
                "column_name": column_name,
                "cells": cells_str
            })
        schema.append({
            "table_name": table_name,
            "columns": schema_columns
        })
    return schema


def table2str(schema_columns):
    table = []
    for i in range(-1, len(schema_columns[0]["cells"])):
        row = []
        for column in schema_columns:
            if i == -1:
                row.append(column["column_name"])
            else:
                try:
                    cell = column["cells"][i]
                except:
                    cell = "NULL"
                row.append(cell)
        row_str = " | ".join(row)
        table.append(row_str)
    table_str = "\n".join(table)
    return table_str

def get_table():
    db_ids = []
    interpretation_dbs = load_json("data/column/disambiguate with DQ/validation_interpretation_db.json")
    for data in tqdm(interpretation_dbs):
        column_interpretation = data["column_interpretation"]
        column_interpretation_new = {}
        table_names = []
        for k in column_interpretation:
            c = column_interpretation[k]
            table_name = ""
            tmp = re.findall("In table \"(.*?)\".", c)
            if len(tmp) > 0:
                table_name = tmp[0]
            else:
                continue
            if table_names.__contains__(table_name):
                column_interpretation_new[table_name][k] = c
            else:
                column_interpretation_new[table_name] = {}
                column_interpretation_new[table_name][k] = c
                table_names.append(table_name)
        data["column_interpretation"] = column_interpretation_new
    write_to_json(interpretation_dbs, "data/column/disambiguate with DQ/validation_interpretation_db_2.json")

def str2context(schema_with_content):
    tables = schema_with_content.split("|")
    schema = ""
    for i, table in enumerate(tables):
        table_name = table.split(":")[0].strip()
        column_with_content = table[table.index(":") + 1:].strip()
        schema += f"[{i+1}] {table_name}\n{column_with_content}\n"
    return schema

def flatten_interpretation(column_interpretation_all):
    out_interpretation = []
    for table in column_interpretation_all.values():
        out_interpretation += list(table.values())
    return out_interpretation

def generate_combinations(input_dict, keys=None):
    if keys is None:
        keys = list(input_dict.keys())
    if len(keys) == 1:
        key = keys[0]
        value_list = []
        for k, v in input_dict[key].items():
            if type(v) == str:
                value_list += [f"\"{key}\": \"{k}\".\"{v}\""]
            else:
                value_list += [f"\"{key}\": \"{k}\".\"{vv}\"" for vv in v]
        return value_list
    else:
        current_key = keys[0]
        rest_keys = keys[1:]
        rest_combinations = generate_combinations(input_dict, rest_keys)
        current_combinations = []
        for k, v in input_dict[current_key].items():
            if type(v) == str:
                current_combinations += [f"\"{current_key}\": \"{k}\".\"{v}\"" ]
            else:
                current_combinations += [f"\"{current_key}\": \"{k}\".\"{vv}\"" for vv in v]
        result = []
        for comb1 in current_combinations:
            for comb2 in rest_combinations:
                result.append(comb1 + " , " + comb2)
        return result

def get_filter_map(ambiguity_map, dq_tmp):
    filter_map = {}
    ambiguity_str = re.findall("(.*?) \((.*)\)$", dq_tmp)
    if len(ambiguity_str) != 0:
        ambiguity_list = ambiguity_str[0][1].split(",")
        for ambiguity in ambiguity_list:
            key, value = re.findall("\"(.*)\": (.*)", ambiguity)[0]
            t, c = re.findall("\"(.*)\".\"(.*)\"", value.strip())[0]
            column = ambiguity_map[key][t]
            if type(column) == list:
                column.remove(c)
            else:
                del ambiguity_map[key][t]
            for table in ambiguity_map[key]:
                if not table in filter_map:
                    filter_map[table] = []
                v = ambiguity_map[key][table]
                if type(v) == list:
                    filter_map[table] += v
                else:
                    filter_map[table] += [v]
    return filter_map

def schema_without_content_to_map(schema_without_content):
    if not schema_without_content:
        return {}
    schema_map = {}
    for table in schema_without_content.split(" | "):
        table_name = table.split(":")[0].strip()
        schema_map[table_name] = []
        columns_str = table[table.index(":") + 1:]
        for column in re.split(r',\s*(?![^(}]*\))', columns_str):
            schema_map[table_name].append(column.strip("\"").strip())
    return schema_map
def schema_from_map(schema_map):
    table_list = []
    for table_name in schema_map:
        column_str = " , ".join(schema_map[table_name])
        table_list.append(f"{table_name} : {column_str}")
    return " | ".join(table_list)


import copy

def get_pooling_selection(sql_dqs, k):
    predict_sqls = []
    while len(predict_sqls) < k and len(sql_dqs) != 0:
        for sql_list in sql_dqs:
            if len(sql_list) == 0:
                sql_dqs.remove(sql_list)
            else:
                sql = sql_list.pop(0)
                if not sql in predict_sqls:
                    predict_sqls.append(sql)
    return predict_sqls[:k]

def filter_top_k_predict_sqls(result_path, k=5):
    result = load_json(result_path)
    for data in result:
        sql_dqs = copy.deepcopy(data["predict_sqls"])
        data["predict_sqls_tmp"] = get_pooling_selection(sql_dqs, k)
    write_to_json(result, result_path)

def tokenize_question(nlp, sentence, attachment_list=None):
    tokens_tmp = nlp.word_tokenize(sentence)
    tokens = []
    jump_flag = 0
    for i in range(len(tokens_tmp)):
        if jump_flag == 1:
            jump_flag = 0
            continue
        token = tokens_tmp[i]
        if token.startswith("#"):
            tokens[-1] += token.strip("#")
        elif token in attachment_list:
            tokens[-1] += token + tokens_tmp[i + 1]
            jump_flag = 1
        else:
            tokens.append(token)
    return tokens

def filter_ddl(ddl_query, columns_filter):
    ddl_query = ddl_query.replace("\n"," ")
    parsed = sqlparse.parse(ddl_query)[0]
    table_name = re.findall("CREATE TABLE(.*?)\(", ddl_query)[0].strip()
    column_str = ""
    for token in parsed.tokens:
        if isinstance(token, Parenthesis):
            column_str += token.value
    column_str = column_str[1:-1]
    columns = column_str.split(",")
    columns_new = []
    for column in columns:
        column = column.strip()
        try:
            column_name = re.findall("`(.*?)`", column)[0]
        except:
            column_name = column.split()[0]
        if column_name in columns_filter or column.startswith("foreign key"):
            columns_new.append(column)
    column_str_new = ', \n'.join(columns_new)
    ddl_query_new = f"CREATE TABLE {table_name}\n (\n{column_str_new}\n)"
    return ddl_query_new

def cluster_self_ambiguous_columns(retriever, columns):
    threshold = 0.66
    clusters = []
    if len(columns) == 2:
        sim = retriever.get_similarity_score(columns[0], columns[1])
        if sim > threshold:
            return columns
        else:
            return []
    while columns:
        # 找出相似度最高的一对元素
        max_similarity = 0
        max_pair = (0,0)
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                sim = retriever.get_similarity_score(columns[i], columns[j])
                if sim > max_similarity:
                    max_similarity = sim
                    max_pair = (i, j)
        # 创建一个新的聚类，并将相似度超过阈值的元素添加到聚类中
        new_cluster = [columns[max_pair[0]]]
        columns = [x for i, x in enumerate(columns) if i != max_pair[0]]
        i = 0
        while i < len(columns):
            if retriever.get_similarity_score(new_cluster[0], columns[i]) > threshold:
                new_cluster.append(columns[i])
                columns = [x for j, x in enumerate(columns) if j != i]
            else:
                i += 1
        clusters.append(new_cluster)
    if len(clusters[0]) >= 2:
        return clusters[0]
    return []

def get_best_result(res_list):
    if isinstance(res_list, str):
        return res_list
    best_res = res_list[0]
    best_num = 0
    res_map = {}
    for res in res_list:
        if not res in res_map:
            res_map[res] = 1
        else:
            res_map[res] += 1
        if res_map[res] >= best_num:
            best_res = res
            best_num = res_map[res]
    return best_res


import sqlite3


def get_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        results = {}
        for table in tables:
            table_name = table[0]
            if table_name.lower() in ["sqlite_sequence", "sqlite_master"]:
                continue
            results[table_name] = {}
            cursor.execute(f"PRAGMA table_info(\"{table_name}\")")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            results[table_name]['columns'] = column_names
            rows_data = []
            for column_name in column_names:
                sql = f"SELECT \"{column_name}\" FROM \"{table_name}\" WHERE \"{column_name}\" IS NOT NULL ORDER BY RANDOM() LIMIT 3"
                cursor.execute(sql)
                rows = cursor.fetchall()
                rows_data.append((column_name, [row[0] for row in rows]))
            results[table_name]['rows'] = rows_data

        schema_with_content = ""
        schema_without_content = ""
        for table_name, data in results.items():
            schema_with_content += f"{table_name} : "
            schema_without_content += f"{table_name} : "
            for column_data in data['rows']:
                column_name = column_data[0]
                rows = column_data[1]
                schema_with_content += f"{column_name} ("
                values = []
                for row in rows:
                    values.append(repr(row))
                schema_with_content += ", ".join(values)
                schema_with_content += "), "
                schema_without_content += f"{column_name}, "

            schema_with_content = schema_with_content.rstrip(", ")
            schema_with_content += " | "
            schema_without_content = schema_without_content.rstrip(", ")
            schema_without_content += " | "

        schema_with_content = schema_with_content.rstrip(" | ")
        schema_without_content = schema_without_content.rstrip(" | ")

        return schema_with_content, schema_without_content
    except sqlite3.Error as e:
        conn.rollback()
        print(e)
    finally:
        cursor.close()
        conn.close()

