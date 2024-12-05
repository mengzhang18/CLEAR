import csv
import os
import random
import re
import sqlite3

from tqdm import tqdm

from src.config import DatasetEnum
from src.prompt_tool import get_prompt_column_interpretation
from src.llm import ask_llm
from src.util import load_json, write_to_json, parse_schema_with_content_to_map


class PreProcessor:

    def __init__(self, args):
        self.dev_path = args.dev_path
        self.dev_process_path = args.dev_process_path
        self.db_root_path = args.db_root_path
        self.dataset_name = args.dataset_name
        self.column_interpretation_path = args.column_interpretation_path
        self.llm_model = args.llm_model
        self.db_primary_keys_path = args.db_primary_keys_path

    def preprocess(self):
        print("========start preprocessing========")

        if not os.path.exists(self.dev_process_path):
            preprocess_dataset = getattr(self, "preprocess_" + self.dataset_name, "This dataset is not supported now!")
            preprocess_dataset()
        if not os.path.exists(self.column_interpretation_path):
            self.column_enrichment()
        if not os.path.exists(self.db_primary_keys_path):
            self.record_primary_keys()

    def preprocess_clambsql(self):
        write_to_json(load_json(self.dev_path), self.dev_process_path)

    def preprocess_ambrosia(self):
        data_ext = []
        with open(self.dev_path, encoding='utf-8') as f:
            csv_dataset = list(csv.reader(f))[1:]
            i = 0
            for data in tqdm(csv_dataset):
                is_ambiguous = data[12]
                if is_ambiguous == "True":
                    db_file = data[11]
                    db_path = os.path.join(self.db_root_path, db_file)
                    schema_with_content, schema_without_content = get_schema(db_path)
                    sqls = [" ".join(sql.replace("\n", " ").split()) for sql in data[2].split("\n\n")]
                    data_ext.append({
                        "index": i,
                        "org_index": int(data[0]),
                        "db_id": "/".join(db_file.split("/")[2:4]),
                        "domain": data[9],
                        "db_file": db_file,
                        "is_ambiguous": data[12],
                        "ambig_type": data[4],
                        "split": data[13],
                        "question": data[1],
                        "gold_queries": sqls,
                        "schema_without_content": schema_without_content,
                        "schema_with_content": schema_with_content
                    })
                    i += 1
        write_to_json(data_ext, self.dev_process_path)

    def preprocess_ambiqt(self):
        data_org = load_json(self.dev_path)
        keywords = {
            "extra_map": {"ambiqt_type": "column", "keys": ["extra_map"]},
            "extra_table_map": {"ambiqt_type": "table", "keys": ["extra_table_map"]},
            "split_map": {"ambiqt_type": "join", "keys": ["primary_key", "split_map"]},
            "all_raw_cols": {"ambiqt_type": "aggregate", "keys": ["all_raw_cols", "all_cols", "new_table_name", "tables_with_pkeys"]}
        }
        data_ext = []
        for i, data in enumerate(data_org):
            data_ext.append({
                "index": i,
                "db_id": data["db_id"]
            })
            keyword_map = {}
            for keyword in keywords:
                if keyword in data:
                    for key in keywords[keyword]["keys"]:
                        keyword_map[key] = data[key]
                    keyword_map["ambiqt_type"] = keywords[keyword]["ambiqt_type"]
                    break
            keyword_map.update({
                "question": data["question"],
                "orig_query": data["orig_query"],
                "query1": data["query1"],
                "query2": data["query2"],
                "schema_without_content": data["schema_without_content"],
                "schema_with_content": data["schema_with_content"]
            })
            data_ext[i].update(keyword_map)
        random.seed(42)
        n_sample = len(data_ext) if len(data_ext) < 400 else 400
        data_ext = random.sample(data_ext, n_sample)
        write_to_json(data_ext, self.dev_process_path)

    def preprocess_bird(self):
        data_org = load_json(self.dev_path)
        data_ext = []
        schema_cache = {}
        for i, qt in enumerate(tqdm(data_org)):
            db_id = qt["db_id"]
            if db_id in schema_cache:
                schema_with_content = schema_cache[db_id]["schema_with_content"]
                schema_without_content = schema_cache[db_id]["schema_without_content"]
            else:
                db_path = os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")
                schema_with_content, schema_without_content = get_schema(db_path)
                schema_cache[db_id] = {}
                schema_cache[db_id]["schema_with_content"] = schema_with_content
                schema_cache[db_id]["schema_without_content"] = schema_without_content
            data_ext.append({
                "index": i,
                "db_id": db_id,
                "question": qt["question"],
                "evidence": qt["evidence"],
                "query": qt["SQL"],
                "difficulty": qt["difficulty"],
                "schema_without_content": schema_without_content,
                "schema_with_content": schema_with_content
            })
        write_to_json(data_ext, self.dev_process_path)

    def preprocess_spider(self):
        data_org = load_json(self.dev_path)
        data_ext = []
        schema_cache = {}
        for i, qt in enumerate(tqdm(data_org)):
            db_id = qt["db_id"]
            if db_id in schema_cache:
                schema_with_content = schema_cache[db_id]["schema_with_content"]
                schema_without_content = schema_cache[db_id]["schema_without_content"]
            else:
                db_path = os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")
                schema_with_content, schema_without_content = get_schema(db_path)
                schema_cache[db_id] = {}
                schema_cache[db_id]["schema_with_content"] = schema_with_content
                schema_cache[db_id]["schema_without_content"] = schema_without_content
            data_ext.append({
                "index": i,
                "db_id": db_id,
                "question": qt["question"],
                "query": qt["query"],
                "schema_without_content": schema_without_content,
                "schema_with_content": schema_with_content
            })
        write_to_json(data_ext, self.dev_process_path)

    def column_enrichment(self):
        dataset = load_json(self.dev_process_path)
        interpretation_caches = {}
        if os.path.exists(self.column_interpretation_path):
            interpretation_caches = load_json(self.column_interpretation_path)
        for i, data in enumerate(tqdm(dataset)):
            if data.__contains__("column_interpretation_all"):
                continue
            schema_with_content = data["schema_with_content"]
            db_id = data["db_id"]
            column_interpretation_cache = {}
            if db_id in interpretation_caches:
                column_interpretation_cache = interpretation_caches[db_id]
            flag_new_db = 0
            if len(column_interpretation_cache) == 0:
                flag_new_db = 1
            schema_with_content_filter = filter_schema_with_content(schema_with_content,
                                                                    column_interpretation_cache)
            while schema_with_content_filter != "":
                prompt = get_prompt_column_interpretation(schema_with_content_filter)
                res = ask_llm(self.llm_model, prompt)
                interpretation_filter = parse_interpretation(res)
                for table in interpretation_filter:
                    if column_interpretation_cache.__contains__(table):
                        column_interpretation_cache[table].update(interpretation_filter[table])
                    else:
                        column_interpretation_cache[table] = interpretation_filter[table]
                if flag_new_db == 1:
                    interpretation_caches[db_id] = column_interpretation_cache
                write_to_json(interpretation_caches, self.column_interpretation_path)
                schema_with_content_filter = filter_schema_with_content(schema_with_content,
                                                                        column_interpretation_cache)
            interpretation_all = get_current_interpretation(schema_with_content, column_interpretation_cache)
            data["column_interpretation_all"] = interpretation_all
            write_to_json(dataset, self.dev_process_path)

    def record_primary_keys(self, lower=False):
        db_primary_keys = {}
        for data in load_json(self.dev_process_path):
            db_id = data["db_id"]
            if db_id in db_primary_keys:
                continue
            if self.dataset_name == DatasetEnum.CLAMBSQL:
                db_path = get_db_path(self.dataset_name, self.db_root_path, data["db_file"])
            else:
                db_path = get_db_path(self.dataset_name, self.db_root_path, db_id)
            primary_key_map = get_primary_keys(db_path)
            if lower:
                primary_key_map_tmp = {}
                for table in primary_key_map:
                    primary_key_map_tmp[table.lower()] = [column.lower() for column in primary_key_map[table]]
                primary_key_map = primary_key_map_tmp
            db_primary_keys[db_id] = primary_key_map
        write_to_json(db_primary_keys, self.db_primary_keys_path)

def get_db_path(dataset_name, db_root_path, db_id):
    if dataset_name in [DatasetEnum.BIRD, DatasetEnum.SPIDER]:
        db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")
    elif dataset_name == DatasetEnum.AMBIQT:
        db_id = db_id.split("/")[1]
        db_path = os.path.join(db_root_path, db_id, f"{db_id}.sqlite")
    elif dataset_name == DatasetEnum.AMBROSIA:
        db_name = db_id.split("/")[1]
        ambig_type = db_name.split("_")[0]
        db_file = f"data/{ambig_type}/{db_id}/{db_name}.sqlite"
        db_path = os.path.join(db_root_path, db_file)
    elif dataset_name == DatasetEnum.CLAMBSQL:
        db_file = db_id
        db_path = os.path.join(db_root_path, db_file)
    else:
        raise Exception("Unknown dataset")
    return db_path

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
                    values.append(repr(row).replace("|",""))
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

def parse_interpretation(res):
    res_list = res.split("\n")
    interpretation = {}
    table_name = ""
    for r in res_list:
        if r.startswith("- "):
            r = r.strip("- ")
        if len(r) == 0:
            continue
        if re.match("^\[\d]", r.strip()) is not None:
            table_name = r.strip().split()[1].split(":")[0]
            table_name = table_name.strip().strip("-").strip("*").strip()
            if table_name:
                interpretation[table_name] = {}
        else:
            column_name = r.split(":")[0].strip()
            column_name = column_name.strip().strip("-").strip("*").strip()
            if table_name and column_name:
                tmp = re.findall("(.*): In table \"(.*)\"\. (.*)", r)
                if tmp and len(tmp[0]) == 3:
                    interpretation[table_name][column_name] = r.strip()
    return interpretation


def filter_schema_with_content(schema_with_content, column_interpretation: dict):
    schema = parse_schema_with_content_to_map(schema_with_content)
    schema_new = []
    for table in schema:
        table_name = table["table_name"]
        columns = table["columns"]
        if column_interpretation.__contains__(table_name) is False:
            schema_new.append(table)
            continue
        columns_new = []
        for column in columns:
            column_name = column["column_name"]
            if column_interpretation[table_name].__contains__(column_name) is False:
                columns_new.append(column)
        if len(columns_new) > 0:
            schema_new.append({"table_name": table_name, "columns": columns_new})
    schema_str_new = ""
    for table in schema_new:
        table_name = table["table_name"]
        columns = table["columns"]
        table_str = table_name + " : "
        for column in columns:
            column_name = column["column_name"]
            cells = column["cells"]
            table_str += "{} {} , ".format(column_name, cells)
        table_str = table_str[:-3]
        schema_str_new += table_str + " | "
    schema_str_new = schema_str_new[:-3]
    return schema_str_new


def get_current_interpretation(schema_with_content, column_interpretation_db):
    column_interpretation_all = {}
    schema = parse_schema_with_content_to_map(schema_with_content)
    for table in schema:
        table_name = table["table_name"]
        columns = table["columns"]
        if column_interpretation_all.__contains__(table_name) is False:
            column_interpretation_all[table_name] = {}
        for column in columns:
            column_name = column["column_name"]
            if column_interpretation_db.__contains__(table_name):
                if column_interpretation_db[table_name].__contains__(column_name):
                    column_interpretation_all[table_name][column_name] = column_interpretation_db[table_name][
                        column_name]
    return column_interpretation_all

def get_primary_keys(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    primary_key_map = {}
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table in tables:
            if table[0] in ["sqlite_sequence", "sqlite_master"]:
                continue
            cursor.execute("PRAGMA table_info(`{}`);".format(table[0]))
            rows = cursor.fetchall()
            primary_keys = [row[1] for row in rows if row[5] == 1]
            primary_key_map[table[0]] = primary_keys
    except Exception as e:
        print(e)
    finally:
        cursor.close()
        conn.close()
    return primary_key_map
