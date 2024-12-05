import json
import os
import sqlite3
from collections import Counter

from tqdm import tqdm

from src.util import load_json, write_to_json


def compare_query_results(predicted_results, gold_results, order_by=False):
    if not predicted_results:
        return False

    if order_by:
        if len(gold_results) != len(predicted_results):
            return False

        if any(len(row) != len(gold_results[0]) for row in gold_results + predicted_results):
            return False

        for gold_row, predicted_row in zip(gold_results, predicted_results):
            if tuple(sort_with_different_types(gold_row)) != tuple(sort_with_different_types(predicted_row)):
                return False
        return True
    else:
        flat_gold = Counter(item for row in gold_results for item in row)
        flat_predicted = Counter(item for row in predicted_results for item in row)

        return flat_gold == flat_predicted


def sort_with_different_types(arr):
    sorted_arr = sorted(arr, key=sort_key)
    return sorted_arr


def sort_key(x):
    if x is None:
        return (0, '')  # Treat None as the smallest value
    elif isinstance(x, (int, float)):
        return (1, float(x))  # Handle numerical types uniformly
    else:
        return (2, str(x))  # Convert all other types to string for consistent comparison

def parse_ambiguity_map(ambiguity_map_str):
    if isinstance(ambiguity_map_str, str):
        return json.loads(ambiguity_map_str)
    elif isinstance(ambiguity_map_str, dict):
        ambiguity_map = {}
        ambiguity_map["query"] = json.loads(ambiguity_map_str["query"])
        ambiguity_map["match"] = json.loads(ambiguity_map_str["match"])
        return ambiguity_map

def execute_sql(predicted_sql, ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    try:
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        res = 0
        if 'order by' in ground_truth.lower():
            is_same = compare_query_results(predicted_res, ground_truth_res, order_by=True)
        else:
            is_same = compare_query_results(predicted_res, ground_truth_res, order_by=False)
        if is_same:
            res = 1
        # if set(predicted_res) == set(ground_truth_res):
        #     res = 1  # bird success
        # predicted_res_set = [set(exec_res) for exec_res in predicted_res]
        # ground_truth_res_set = [set(exec_res) for exec_res in ground_truth_res]
        # if predicted_res == ground_truth_res:
        #     res = 1  # restrict success
        # elif set(predicted_res) == set(ground_truth_res):
        #     res = 1  # bird success
        # elif predicted_res_set == ground_truth_res_set:
        #     res = 1  # no_order
    except Exception as e:
        # print(e)
        res = 0
    finally:
        cursor.close()
        conn.close()
    return res


def evaluate_EX(result_path, db_root_path):
    result_types = {"query": [], "match": []}
    match = ["column", "table", "join", "aggregate", "vague"]
    for data in load_json(result_path):
        if data["ambig_type"] in match:
            result_types["match"].append(data)
        else:
            result_types["query"].append(data)
    eval_all = {}
    res_index = []
    for _type in result_types:
        res_list = []
        for data in result_types[_type]:
            # print(data["index"])
            pred_sql = data["predict_sql"]
            gold_sql = data["gold_query"]
            res = 1
            if data["ambig_type"] in match:
                # continue
                clear_ambiguity = parse_ambiguity_map(data["clear_ambiguity"])
                for token in clear_ambiguity:
                    for table in clear_ambiguity[token]:
                        if table.lower() not in pred_sql.lower():
                            res = 0
                            break
                        for column in clear_ambiguity[token][table]:
                            if column.lower() not in pred_sql.lower():
                                res = 0
                                break
            if res == 1:
                db_path = os.path.join(db_root_path, data["db_file"])
                res = execute_sql(pred_sql, gold_sql, db_path)
            res_list.append(res)
        eval_all[_type] = sum(res_list) / len(res_list) * 100 if res_list else 0
    eval_all["overall"] = (eval_all["query"] * 400 + eval_all["match"] * 1000) / 1400
    # print("=====================================")
    # print("       ".join([*eval_all]))
    # print("================ EX =================")
    print("       ".join([str(eval_all[_type])[:5] for _type in eval_all]))


def evaluate_EM(result_path, db_root_path):
    #  TODO
    pass


def evaluate_MM(result_path):
    result = load_json(result_path)
    precision = []
    recall = []
    for data in result:
        gold_ambiguity = parse_ambiguity_map(data["gold_ambiguity"])
        pred_ambiguity = parse_ambiguity_map(data["ambiguity_mapping"])
        scores = []
        for key in gold_ambiguity:
            for token2 in pred_ambiguity[key]:
                table_dicts2 = pred_ambiguity[key][token2]
                sub_scores = []
                for token1 in gold_ambiguity[key]:
                    table_dicts1 = gold_ambiguity[key][token1]
                    try:
                        intersection_table = [element for element in table_dicts1 if element in table_dicts2]
                    except:
                        intersection_table = []
                    score = int(len(intersection_table) == len(table_dicts1))
                    sub_scores.append(score)
                max_score = max(sub_scores) if sub_scores else 0
                scores.append(max_score)
        score = sum(scores) / len(scores) if scores else 0
        precision.append(score)
        scores = []
        for key in gold_ambiguity:
            for token1 in gold_ambiguity[key]:
                table_dicts1 = gold_ambiguity[key][token1]
                sub_scores = []
                for token2 in pred_ambiguity[key]:
                    table_dicts2 = pred_ambiguity[key][token2]
                    try:
                        intersection_table = [element for element in table_dicts1 if element in table_dicts2]
                    except:
                        intersection_table = []
                    score = int(len(intersection_table) == len(table_dicts1))
                    sub_scores.append(score)
                max_score = max(sub_scores) if sub_scores else 0
                scores.append(max_score)
        score = sum(scores) / len(scores) if scores else 0
        recall.append(score)
    p = sum(precision) / len(precision) * 100
    r = sum(recall) / len(recall) * 100
    print("================ MM =================")
    print("       {}       {}      ".format("recall", "precision"))
    print("       {:.1f}        {:.1f}   ".format(r, p))


if __name__ == '__main__':
    db_root_path = "../../dataset/database"
    result_path = "./result_llama3_70b_base.json"

    # evaluate_EX(result_path, db_root_path)
    #
    # evaluate_MM("./clear/mini/result_mapping.json")
    # evaluate_MM("./clear/mini/result_mapping_1.json")
    # evaluate_MM("./clear/3.5/result_mapping.json")
    # evaluate_MM("./clear/result_mapping_vague.json")
    # evaluate_MM("./clear/result_mapping_revision.json")

    dir_path = "./"
    for file in os.listdir(dir_path):
        if file.startswith("result_"):
            # print(file)
            result_path = os.path.join(dir_path, file)
            evaluate_EX(result_path, db_root_path)

    # base = load_json("./result_llama3_70b_base.json")
    # result = load_json("./result_llama3_70b_clear.json")
    # out = []
    # for data1,data2 in zip(base, result):
    #     if data1["ambig_type"] in ["scope", "attachment"]:
    #         # db_path = os.path.join(db_root_path, data1["db_file"])
    #         # res1 = execute_sql(data1["predict_sql"], data1["gold_query"], db_path)
    #         # res2 = execute_sql(data2["predict_sql"], data2["gold_query"], db_path)
    #         # if res1 == 1 and res2 == 0:
    #         #     print(data1["index"])
    #         #     data2["predict_sql"] = data1["predict_sql"]
    #         if data2["ambiguity_clarification"] == "{}":
    #             data2["predict_sql"] = data1["predict_sql"]
    #     out.append(data2)
    # write_to_json(result, "./result_llama3_70b_clear.json")
    # evaluate_EX("./result_llama3_70b_clear.json", db_root_path)

    # result = load_json("./result_llama3_70b_beam.json")
    # s_count = 0
    # for data in result:
    #     if data["ambig_type"] in ["scope", "attachment"]:
    #         db_path = os.path.join(db_root_path, data["db_file"])
    #         for sql in data["predict_sqls"]:
    #             res = execute_sql(sql, data["gold_query"], db_path)
    #             if res == 1:
    #                 s_count += 1
    #                 break
    # print(s_count / 400 * 100)

    # dir_path = "../clambsql/clear"
    # for file in os.listdir(dir_path):
    #     if file.startswith("result_"):
    #         dataset = load_json("../../dataset/clambsql.json")
    #         result = load_json(os.path.join(dir_path, file))
    #         out = []
    #         for data1, data2 in zip(dataset, result):
    #             data2["gold_ambiguity"] = data1["gold_ambiguity"]
    #             data2["clarification_context"] = data1["clarification_context"]
    #             data2["clear_ambiguity"] = data1["clear_ambiguity"]
    #             data2["gold_query"] = data1["gold_query"]
    #             out.append(data2)
    #         write_to_json(out, os.path.join(dir_path, file))

    # clarification = load_json("./clear/result_clarification.json")
    # mapping = load_json("./clear/result_mapping_revision.json")
    # for data1,data2 in zip(mapping, clarification):
    #     for token in data2["msg"]:
    #         if data2["msg"][token]["type"] == 1:
    #             content = parse_ambiguity_map(data2["msg"][token]["content"])
    #             ambiguity_map = parse_ambiguity_map(data1["ambiguity_mapping"])["match"]
    #             if token in ambiguity_map:
    #                 ambiguity_map[token].append(content[token])
    #             data1["ambiguity_mapping"]["match"] = json.dumps(ambiguity_map)
    # write_to_json(mapping, "./clear/result_mapping_revision.json")
