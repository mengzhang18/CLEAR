import shutil

from src.mapping_tool import *
from src.util import *
import copy
import itertools
from stanfordcorenlp import StanfordCoreNLP


class Rewriter:
    def __init__(self, input_path, output_path, ambiqt_type=None, with_content=True):
        self.input_path = input_path
        self.output_path = output_path
        self.with_content = with_content
        self.ambiqt_type = ambiqt_type
        self.max_dq_num = 4
        self.nlp = StanfordCoreNLP(r'E:/ubuntu/ambiguity/stanford-corenlp-4.5.6')

    def rewrite_ambiguity(self, mode="all"):
        assert self.input_path and self.output_path
        result = load_json(self.input_path)
        out = []
        print("=====start rewriting=====")
        for data in tqdm(result):
            if "ambiqt_type" in data:
                self.ambiqt_type = data["ambiqt_type"]
            data = self.get_decomposed_data(data, mode)
            data = self.get_rewritten_question(data, mode)
            data = self.get_written_schema(data)
            out.append(data)
        write_to_json(out, self.output_path)

    def rewrite_clarification(self, mode="all"):
        self.max_dq_num = 100
        result = load_json(self.input_path)
        for data in result:
            data["ambiguity_mapping_copy"] = copy.deepcopy(data["ambiguity_mapping"])
            ambiguity_clarification = parse_ambiguity_map(data["ambiguity_clarification"])
            ambiguity_map = parse_ambiguity_map(data["ambiguity_mapping"])
            ambiguity_map_copy = copy.deepcopy(ambiguity_map)
            for key in ambiguity_map_copy:
                for token in ambiguity_map_copy[key]:
                    if token not in ambiguity_clarification:
                        del ambiguity_map[key][token]
                        continue
                    if type(ambiguity_clarification[token]) != dict or key == "query":
                        continue
                    # vagueness by feedback
                    if ambiguity_clarification[token] not in ambiguity_map_copy[key][token]:
                        ambiguity_map[key][token].append(ambiguity_clarification[token])
                ambiguity_map[key] = json.dumps(ambiguity_map[key])
            data["ambiguity_mapping"] = ambiguity_map
        write_to_json(result, self.output_path)
        input_path = self.input_path
        self.input_path = self.output_path
        self.rewrite_ambiguity(mode)
        self.input_path = input_path
        result = load_json(self.output_path)
        out = []
        for data in tqdm(result):
            clarification_map_str = data["ambiguity_clarification"]
            question = data["question"]
            dqs_tmp = data["DQs_tmp"]
            idx = None
            for i in range(len(dqs_tmp)):
                ambiguity_str = dqs_tmp[i].replace(question, "").strip()
                if ambiguity_str:
                    ambiguity_str = ambiguity_str[1:-1]
                else:
                    idx = 0
                    break
                if ambiguity_str == clarification_map_str:
                    idx = i
                    break
            data["dq_tmp"] = data["DQs_tmp"][idx]
            data["dq"] = data["DQs"][idx]
            data["schema_without_content_dq"] = data["schema_without_content_dqs"][idx]
            if self.with_content:
                data["schema_with_content_dq"] = data["schema_with_content_dqs"][idx]
            data["ambiguity_mapping"] = data["ambiguity_mapping_copy"]
            del data["schema_without_content_dqs"]
            if self.with_content:
                del data["schema_with_content_dqs"]
            del data["ambiguity_mapping_copy"]
            del data["DQs_tmp"]
            del data["DQs"]
            out.append(data)
        write_to_json(out, self.output_path)

    def get_decomposed_data(self, data, mode):
        question = data["question"]
        ambiguity_map = parse_ambiguity_map(data["ambiguity_mapping"])
        DQs = []
        if mode in ["query", "all"]:
            query_ambiguity_map = ambiguity_map["query"]
            DQs += [question + f" ({json.dumps({token:a})})" for token in query_ambiguity_map for a in
                               query_ambiguity_map[token]]
        if mode in ["match", "all"]:
            data_ambiguity_map = ambiguity_map["match"]
            if data_ambiguity_map:
                combinations = self.cartesian_product(data_ambiguity_map)
                DQs += [f"{question} ({json.dumps(combination)})" for combination in combinations]
        if not DQs:
            DQs = [question]
        DQs = DQs[:self.max_dq_num]
        if self.ambiqt_type in ["aggregate", "table"]:
            data["DQs_tmp"] = [DQs[0], DQs[-1]]
        else:
            data["DQs_tmp"] = DQs
        return data

    def get_rewritten_question(self, data, mode):
        # TODO extend with query ambiguity rewritten
        data["DQs"] = []
        if mode in ["query", "all"]:
            data1 = self.get_query_rewritten_question(copy.deepcopy(data))
            data["DQs"] += data1["DQs"]
        if mode in ["match", "all"]:
            data2 = self.get_data_rewritten_question(copy.deepcopy(data))
            data["DQs"] += data2["DQs"]
        return data

    def get_query_rewritten_question(self, data):
        question = data["question"]
        ambiguity_map = parse_ambiguity_map(data["ambiguity_mapping"])
        ambiguity_map = ambiguity_map["query"]
        num = 0
        for token in ambiguity_map:
            if num == 0:
                num = len(ambiguity_map[token])
            else:
                num *= len(ambiguity_map[token])
        if data["ambig_type"] in ["scope", "attachment"]:
            dqs = self.generate_scope_dqs(question, ambiguity_map)
        else:
            dqs = [question]*num
        data["DQs"] = dqs
        return data

    def generate_scope_dqs(self, question, ambiguity_map):
        dqs = []
        for token in ambiguity_map:
            for clarify_token in ambiguity_map[token]:
                dq = question.replace(token, clarify_token)
                dqs.append(dq)
        return dqs

    # def generate_attachment_dqs(self, question, ambiguity_map):
    #     dqs = []
    #     ambiguity_map = self.distribute_and_transform(ambiguity_map)
    #     for token in ambiguity_map:
    #         for clarify_token in ambiguity_map[token]:
    #             dq = question.replace(token, clarify_token)
    #             dqs.append(dq)
    #     return dqs

    @staticmethod
    def get_data_ambiguity_map(data):
        try:
            ambiguity_map = parse_ambiguity_map(data["ambiguity_mapping"])
            if "query" in ambiguity_map and "match" in ambiguity_map:
                ambiguity_map = ambiguity_map["match"]
        except:
            ambiguity_map = {}
        return ambiguity_map

    def get_data_rewritten_question(self, data):
        question = data["question"]
        ambiguity_map = self.get_data_ambiguity_map(data)
        question_tokens = tokenize_question(self.nlp, question, ["-"])
        dqs_tmp = data["DQs_tmp"]
        dqs = []
        for dq_tmp in dqs_tmp:
            dq = question
            ambiguity_str = dq_tmp.replace(question, "").strip()
            token_ambiguity_map = {}
            if len(ambiguity_str) != 0:
                ambiguity_str = ambiguity_str[1:-1]
                ambiguity_str_map = json.loads(ambiguity_str)
                if self.is_query_ambiguity(ambiguity_str_map):
                    continue
                for token in ambiguity_str_map:
                    table_dict = ambiguity_str_map[token]
                    table_list = convert_table_dict_to_list(table_dict)
                    rewrite_strategy = 0  # substitude or prompt
                    value = f"{token} ({' and '.join(table_list)})"
                    for table_column in table_list:
                        t = table_column.split(".")[0]
                        c = table_column.split(".")[1]
                        # if t == "_AND_":
                        #     v_and = self.get_and_column(c)
                        #     value = f"{token} ({v_and})"
                        # else:
                        #     if not ambiqt_type:
                        #         if len(ambiguity_map[token]) == 1:
                        #             value = f"{c}"
                        #         else:
                        #             if c.endswith("id"):
                        #                 value = f"{token} (\"{t}\")"
                        #             else:
                        #                 value = f"{token} (\"{t}\".\"{c}\")"
                        #     else:
                        #         value = self.rewrite_ambiqt(token, t, c)
                    token_ambiguity_map[token] = value.strip()
                dq_tokens = []
                for token in question_tokens:
                    if token in token_ambiguity_map:
                        token_tmp = token_ambiguity_map[token]
                        del token_ambiguity_map[token]
                        token = token_tmp
                    dq_tokens.append(token)
                dq = " ".join(dq_tokens)
                for token in token_ambiguity_map:
                    if re.findall(token, question) == re.findall(token, dq) != []:
                        dq = dq.replace(token, token_ambiguity_map[token])
                dq = (dq.replace(" .", ".").replace(" ,", ",").replace(" ?", "?"))
                dq_quote = re.findall("'([^']*)'|\"([^\"]*)\"", dq)
                for quote_tuple in dq_quote:
                    s_quote, d_quote = quote_tuple
                    dq = dq.replace(s_quote, s_quote.strip()).replace(d_quote, d_quote.strip())
            dqs.append(dq)
        data["DQs"] = dqs
        return data

    def rewrite_ambiqt(self, token, t, c):
        value = f"\"{t}\".\"{c}\""
        if self.ambiqt_type in ["column"]:
            value = f"{c}"
        elif self.ambiqt_type in ["table", "join"]:
            if c.endswith("id"):
                value = f"{token} (\"{t}\")"
            else:
                value = f"{token} (\"{t}\".\"{c}\")"
        elif self.ambiqt_type in ["aggregate"]:
            agg_map = {"max": "maximum", "min": "minimum", "avg": "average", "sum": "total"}
            flag = 0
            for agg in agg_map:
                if agg_map[agg] in token:
                    if c.startswith(agg):
                        value = f"{c} (\"{t}\".\"{c}\")"
                    else:
                        value = f"{token} (\"{t}\".\"{c}\")"
                    flag = 1
                    break
            if flag == 0:
                value = f"{c} (\"{t}\".\"{c}\")"
        else:
            value = f"{token}"
        return value

    @staticmethod
    def is_query_ambiguity(ambiguity_str_map):
        for token in ambiguity_str_map:
            value = ambiguity_str_map[token]
            if isinstance(value, str):
                return True
        return False

    @staticmethod
    def get_and_column(and_map):
        s_list = []
        for table in and_map:
            v = and_map[table]
            if isinstance(v, str):
                v = [v]
            for column in v:
                s_list.append(f"\"{table}\".\"{column}\"")
        s = " and ".join(s_list)
        return s

    def get_written_schema(self, data):
        schema_with_content = data["schema_with_content"] if self.with_content else None
        schema_without_content = data["schema_without_content"]
        ambiguity_map = self.get_data_ambiguity_map(data)
        question = data["question"]
        dqs_tmp = data["DQs_tmp"]
        schema_without_content_dqs = []
        schema_with_content_dqs = []
        for dq_tmp in dqs_tmp:
            ambiguity_str = dq_tmp.replace(question, "").strip()[1:-1]
            rm_list = []
            if ambiguity_str:
                ambiguity_str_map = json.loads(ambiguity_str)
                if self.is_query_ambiguity(ambiguity_str_map):
                    schema_without_content_dqs.append(schema_without_content)
                    if self.with_content:
                        schema_with_content_dqs.append(schema_with_content)
                    continue
                ambiguity_map_copy = copy.deepcopy(ambiguity_map)
                for token in ambiguity_map:
                    sub_schema = ambiguity_str_map[token]
                    schema_tuple = ambiguity_map_copy[token]
                    schema_tuple.remove(sub_schema)
                    for table_dict in schema_tuple:
                        for table in table_dict:
                            for column in table_dict[table]:
                                rm_list.append((table, column))
            schema_without_content_map = schema_without_content_to_map(schema_without_content)
            schema_with_content_map = schema_without_content_to_map(schema_with_content) if self.with_content else None
            for rm_tuple in rm_list:
                schema_without_content_map[rm_tuple[0]] = [value for value in schema_without_content_map[rm_tuple[0]] if
                                                           value != rm_tuple[1]]
            schema_without_content_dqs.append(schema_from_map(schema_without_content_map))
            if self.with_content:
                for rm_tuple in rm_list:
                    schema_with_content_map[rm_tuple[0]] = [value for value in schema_with_content_map[rm_tuple[0]] if
                                                            value != rm_tuple[1]]
                schema_with_content_dqs.append(schema_from_map(schema_with_content_map))
        data["schema_without_content_dqs"] = schema_without_content_dqs
        if self.with_content:
            data["schema_with_content_dqs"] = schema_with_content_dqs
        return data

    @staticmethod
    def generate_combinations(input_map):
        keys = input_map.keys()
        list_of_values = []
        for k in keys:
            sub_values = []
            for sub_k, sub_v in input_map[k].items():
                if isinstance(sub_v, list):
                    for value in sub_v:
                        sub_values.append({sub_k: value})
                else:
                    sub_values.append({sub_k: sub_v})
            list_of_values.append(sub_values)
        combinations = list(itertools.product(*list_of_values))
        results = []
        for combo in combinations:
            result = {}
            for i, k in enumerate(keys):
                result[k] = combo[i]
            results.append(result)
        return results

    @staticmethod
    def cartesian_product(input_map):
        keys = input_map.keys()
        values = input_map.values()

        for prod in itertools.product(*values):
            yield dict(zip(keys, prod))

    @staticmethod
    def distribute_and_transform(input_dict):
        output_dict = {}
        for key, value_list in input_dict.items():
            transformed_value_list = []
            for value in value_list:
                if ' and ' in value:
                    and_split = value.rsplit(' and ', 1)
                    a, b = and_split[0], and_split[1]
                    new_key = f"{a} and {b} {key}"
                    transformed_value_list.append(f"{b} {key}, and {a}")
                    transformed_value_list.append(f"{a} {key}, and {b} {key}")
                else:
                    new_key = f"{value} {key}"
            if new_key in output_dict:
                output_dict[new_key] += transformed_value_list
            else:
                output_dict[new_key] = transformed_value_list
        return output_dict


if __name__ == '__main__':
    rewriter = Rewriter(input_path="../data/bird/evidence/result_clarification.json",
                        output_path="../data/bird/evidence/result_rewriting.json", with_content=False)
    rewriter.rewrite_clarification()
