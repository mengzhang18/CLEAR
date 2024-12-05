import argparse
import json
import os
import random
import re
import sqlite3
import time

import sql_metadata
import sqlparse
import torch
from func_timeout import func_timeout
from openai import OpenAI
from tqdm import tqdm
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import T5ForConditionalGeneration, T5Tokenizer, LogitsProcessor

from src.mapping_tool import *

from src.mapping_tool import parse_ambiguity_map
from src.util import get_best_result

load_dotenv()

PROMPT_BASE = '''## Text-to-SQL task
### The task is to write SQL queries based on the provided questions in English. 
### Questions can take the form of an instruction or command. 
### Do not include any explanations, and do not select extra columns beyond those requested in the question.

Given the following SQLite database schema:

[SCHEMA_WITH_CONTENT]

Answer the following:
[QUESTION]

SQL:'''

PROMPT_DETECTION = '''## Text-to-SQL task
### The task is to write SQL queries based on the provided questions in English. 
### Questions can take the form of an instruction or command and can be ambiguous, meaning they can be interpreted in different ways. 
### In such cases, write all possible SQL queries corresponding to different interpretations and separate each SQL query with an empty line.
### Do not include any explanations, and do not select extra columns beyond those requested in the question.

Given the following SQLite database schema:

[SCHEMA_WITH_CONTENT]

Answer the following:
[QUESTION]

SQL:'''


def load_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)


def write_to_json(data, path):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.flush()


def ask_llm(model, prompt, temperature=0.0, max_tokens=2048, stop=None, n=1, api_base=None, api_key=None):
    llm = OpenAI()
    if api_base and api_key:
        llm = OpenAI(base_url=api_base, api_key=api_key)
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{
            "role": "user",
            "content": prompt
        }]
    is_error = True
    res = ""
    t = 1
    while is_error:
        try:
            response = llm.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                n=n
            )
            if isinstance(response, str):
                print(response)
                time.sleep(t)
                t += 1
            else:
                res_list = []
                for choice in response.choices:
                    res = choice.message.content
                    res_list.append(res)
                if n == 1:
                    res = res_list[0]
                else:
                    res = res_list
                is_error = False
        except Exception as e:
            print(e)
            time.sleep(t)
            t += 1
    return res


def generate_vllm(model_name, generator, prompt, mode):
    if mode == "beam":
        completion = generator.completions.create(
            model=model_name,
            prompt=prompt,
            extra_body={"use_beam_search": True, "best_of": 5},
            temperature=0.0,
            n=5,
            max_tokens=500, seed=42)
        outputs = [row.text for row in completion.choices]
    else:
        response = generator.chat.completions.create(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            max_tokens=500,
            seed=42)
        outputs = response.choices[0].message.content
    return outputs


def load_t5_model(checkpt_dir, device):
    model = T5ForConditionalGeneration.from_pretrained(checkpt_dir)
    model.to(device)
    return model


def load_t5_tokenizer(checkpt_dir):
    tokenizer = T5Tokenizer.from_pretrained(checkpt_dir, model_max_length=512)
    return tokenizer


def generate_t5(question, t2s_model, t2s_tokenizer, db_id, schema_without_content, num_outputs=1, beam_width=5):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    t2s_model.eval()
    addendum = " | {} | {}".format(db_id, schema_without_content)
    model_input = " ".join(question.split()).replace(" , ", ", ") + addendum
    # if checkpt_path and 'flan' in checkpt_path:
    model_input = "semantic parse: " + model_input
    encoded = t2s_tokenizer(model_input, max_length=512, truncation=True,
                            return_tensors="pt").input_ids

    with torch.no_grad():
        outputs = t2s_model.generate(encoded.to(device), num_beams=beam_width, num_return_sequences=num_outputs,
                                     max_length=512)
        outputs = t2s_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outputs = [remove_db_prefix_from_sql(o) for o in outputs]
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def remove_db_prefix_from_sql(sql):
    if '|' in sql:
        sql = sql[sql.find('|') + 1:].strip()
    return sql


def convert_text_to_template(question, template_model, template_tokenizer, db_id, schema_without_content, prefix="",
                             num_outputs=1, beam_width=5):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    template_model.eval()
    addendum = " | {} | {}".format(db_id, schema_without_content)
    model_input = "template generation: " + " ".join(question.split()).replace(" , ", ", ") + addendum
    if prefix == "":
        encoded = template_tokenizer(model_input, max_length=512, truncation=True,
                                     return_tensors="pt").input_ids

        with torch.no_grad():
            outputs = template_model.generate(encoded.to(device),
                                              num_beams=beam_width, num_return_sequences=num_outputs,
                                              max_length=512)
        outputs = template_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    else:
        outputs = get_output_with_prefix(template_model, template_tokenizer, model_input, prefix, beam_width=beam_width,
                                         num_outputs=num_outputs)

    outputs = [remove_db_prefix_from_sql(o) for o in outputs]
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def get_output_with_prefix(model, tokenizer, model_input, prefix, beam_width=10, num_outputs=1):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoded = tokenizer(model_input, max_length=512, truncation=True, return_tensors="pt").input_ids
    prefix_encoded = tokenizer(prefix, max_length=512, truncation=True, return_tensors="pt").input_ids[0].tolist()
    idx = prefix_encoded.index(1)
    tokens = prefix_encoded[:idx]
    if tokens[0] != 0:
        tokens = [0] + tokens
    logits_processor = [EnforcePrefixLogitsProcessor(tokens)]
    with torch.no_grad():
        outputs = model.generate(encoded.to(device),
                                 num_beams=beam_width, num_return_sequences=num_outputs,
                                 max_length=512, logits_processor=logits_processor)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs


class EnforcePrefixLogitsProcessor(LogitsProcessor):
    def __init__(self, tokens):
        self.n_tokens = len(tokens)
        self.tokens = tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_index = input_ids.shape[-1]
        if current_index >= self.n_tokens:
            return scores
        idxes = torch.LongTensor([self.tokens[current_index]])
        oh = F.one_hot(idxes, num_classes=scores.shape[1]).to(scores.dtype)
        oh[oh == 0] = -float("inf")
        scores = oh.repeat(scores.shape[0], 1).to(scores.device)
        return scores


def generate_logical_template(question, template_model, template_tokenizer, db_id, schema_without_content,
                              beam_width=5):
    top = convert_text_to_template(question, template_model, template_tokenizer, db_id, schema_without_content,
                                   num_outputs=1, beam_width=beam_width)
    outs = []
    top = top.split('@')
    oj = int(top[0].strip().split(' ')[0])
    osx = int(top[1].strip().split(' ')[0])
    num_outputs = min(beam_width, 3)
    if num_outputs == 1:
        outs.append(
            convert_text_to_template(question, template_model, template_tokenizer, db_id, schema_without_content,
                                     prefix="{} joins @ {} selects @".format(oj, osx), num_outputs=1,
                                     beam_width=beam_width))
    else:
        outs += convert_text_to_template(question, template_model, template_tokenizer, db_id, schema_without_content,
                                         prefix="{} joins @ {} selects @".format(oj, osx), num_outputs=1,
                                         beam_width=beam_width)
    if oj >= 4:
        njs = [oj - 1, oj - 2]
    else:
        njs = [oj - 1, oj + 1]
    for nj in njs:
        if nj >= 0:
            num_outputs = min(beam_width, 3 if oj == 0 else 2)
            if num_outputs == 1:
                outs.append(convert_text_to_template(question, template_model, template_tokenizer, db_id,
                                                     schema_without_content,
                                                     prefix="{} joins @ {} selects @".format(nj, osx), num_outputs=1,
                                                     beam_width=beam_width))
            else:
                outs += convert_text_to_template(question, template_model, template_tokenizer, db_id,
                                                 schema_without_content,
                                                 prefix="{} joins @ {} selects @".format(nj, osx), num_outputs=1,
                                                 beam_width=beam_width)
    for ns in [osx - 1, osx + 1]:
        if ns > 0:
            nout = convert_text_to_template(question, template_model, template_tokenizer, db_id, schema_without_content,
                                            prefix="{} joins @ {} selects @".format(oj, ns), num_outputs=1,
                                            beam_width=beam_width)
            outs.append(nout)
    return outs


def template_fill(question, model, tokenizer, db_id, schema_without_content, templates, beam_width=5, num_outputs=5):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    filled = []
    for template in templates:
        if '@' in template:
            template = template[template.rfind('@') + 1:].strip()
        template = " ".join(template.split())
        model_input = "template fill: {} | {} | {} @ {}".format(question, db_id, schema_without_content, template)
        output_fn = get_output_controlled
        filled.append(output_fn(model, tokenizer=tokenizer,
                                model_input=model_input, template=template,
                                db_id=db_id, column=True, table=True,
                                schema_without_content=schema_without_content, beam_width=beam_width,
                                num_outputs=num_outputs, device=device))
    return filled


def get_output_controlled(model, tokenizer, model_input, template,
                          db_id, column=True, table=False, schema_without_content=None,
                          beam_width=10, num_outputs=1, device=None):
    encoded = tokenizer(model_input, max_length=512, truncation=True,
                        return_tensors="pt").input_ids
    cs_logits_processor = ControlSplitLogitsProcessor(tokenizer, template,
                                                      db_id, column=column, table=table,
                                                      schema_without_content=schema_without_content)
    logits_processor = [cs_logits_processor]
    with torch.no_grad():
        outputs = model.generate(encoded.to(device),
                                 num_beams=beam_width, num_return_sequences=num_outputs,
                                 max_length=512, logits_processor=logits_processor)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outputs = [x[x.find('|') + 1:].strip() if '|' in x else x for x in outputs]
    outputs = [normalize_sql(x) for x in outputs]
    templates = [cs_logits_processor.templatize(x) for x in outputs]
    template_portion = template[template.rfind('@') + 1:].strip()
    outputs = [output for output, ptemplate in zip(outputs, templates) if \
               ptemplate == template_portion]
    outputs = [outputs[i] for i in range(len(outputs)) if outputs[i] not in \
               outputs[:i]]
    return outputs


def extract_sql(sql):
    if '|' in sql:
        sql = sql[sql.find('|') + 1:].strip()
    if '@' in sql:
        sql = sql[sql.find('@') + 1:].strip()
    return sql


class ControlSplitLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, template, db_id, column=True, table=False, \
                 schema_without_content=None):
        self.tokenizer = tokenizer
        self.template = template
        self.db_id = db_id
        self.column = column
        self.table = table
        self.tables, self.columns = extract_tables_and_columns(
            schema_without_content)
        self.templatize = lambda sql: templatize_sql_from_map(
            extract_sql(sql), self.tables, self.columns)
        self.table_tokens = [self.tokenizer.encode(table)[0] for \
                             table in self.tables]
        self.column_tokens = [self.tokenizer.encode(column)[0] for \
                              column in self.columns + ['*']]
        self.column_tokens += [self.tokenizer.encode("t1." + column)[3] for \
                               column in self.columns]
        self.allowed_tokens = []
        if column:
            self.allowed_tokens += self.column_tokens
        if table:
            self.allowed_tokens += self.table_tokens
        self.allowed_tokens = torch.LongTensor(list(set(
            self.allowed_tokens)))

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] >= 128:
            return scores

        decoded = self.tokenizer.batch_decode(input_ids,
                                              skip_special_tokens=True, clean_up_tokenization_spaces=True)
        disallowed_positions = torch.Tensor([disallowed(self.template, d,
                                                        self.templatize(d), column=self.column, table=self.table) for \
                                             d in decoded]).unsqueeze(1)
        idxes = torch.argmax(scores, dim=1)
        oh = F.one_hot(idxes, num_classes=scores.shape[1]). \
            to(scores.dtype).to(scores.device)
        oh[oh == 0] = -float("inf")

        scores_disallowed = oh.to(scores.device)
        disallowed_positions = disallowed_positions.to(scores.device)
        disallowed_positions = disallowed_positions.repeat(1, scores.shape[1])
        scores[disallowed_positions == 1] = scores_disallowed[disallowed_positions == 1]
        scores[disallowed_positions == -1] = -float("inf")

        if self.allowed_tokens is not None:
            aoh = F.one_hot(self.allowed_tokens,
                            num_classes=scores.shape[1]).to(scores.dtype).to(scores.device)
            aoh = torch.sum(aoh, axis=0).unsqueeze(0).repeat(scores.shape[0], 1)
            scores[(disallowed_positions + aoh) == 0] = -float("inf")

        return scores


def templatize_sql_from_map(sql, tables, columns):
    sql = normalize_sql(sql)
    sql = sql.replace("''", "\"").replace(",", " , "). \
        replace("(", " ( ").replace(")", " ) ").split()
    template = []
    current_quote = None
    for token in sql:
        if current_quote is not None:
            if token[-1] == current_quote:
                current_quote = None
                template.append("string")
        else:
            if token[0] in ['\'', '"', '`']:
                if len(token) > 1 and token[-1] == token[0]:
                    template.append("string")
                else:
                    current_quote = token[0]
            elif token.isnumeric():
                template.append("number")
            else:
                if len(token) > 3 and token[0] in ('t', 'T') and token[2] == '.':
                    token_pref = token[3:].lower()
                else:
                    token_pref = token.lower()
                if token_pref in tables:
                    template.append("table")
                elif token_pref in columns or token_pref == '*':
                    template.append("column")
                else:
                    template.append(token)
    template = " ".join(template)
    template = template.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    template = template.lower()
    for kword in ["count", "avg", "sum", "min", "max"]:
        template = template.replace(kword + " (", kword + "(")
    template = template.replace(") ,", "),")
    return normalize_sql(template)


def extract_tables_and_columns(schema):
    # Without content
    tables = []
    columns = []
    for tbl_string in schema.strip().split('|'):
        tbl_string = tbl_string.split(":")
        tables.append(tbl_string[0].strip().lower())
        for col_string in tbl_string[1].strip().split(','):
            columns.append(col_string.strip().lower())
    return tables, columns


def disallowed(template, output, template_pred, column=True, table=False,
               enforce_adherence=True):
    if "|" in output:
        output = output[output.rfind('|') + 1:].strip()
    if "@" in output:
        output = output[output.rfind('@') + 1:].strip()
    template_pred_previous = "" if " " not in template_pred else \
        template_pred[:template_pred.rfind(" ")]
    if enforce_adherence and not template.startswith(template_pred_previous):
        return -1
    if ' join ' in template and len(output) > 0:
        last_token = output.split(" ")[-1].lower()
        if column and (len(last_token) == 3 and last_token.startswith("t") and
                       last_token[2] == '.'):
            return 0
    if template.startswith(template_pred) and len(template_pred) > 0:
        last_token = output.split(" ")[-1].lower()
        portion = template[len(template_pred):].strip()
        if ' join ' in template:
            if column and (len(last_token) == 3 and last_token.startswith("t") and
                           last_token[2] == '.'):
                return 0
        else:
            if column and portion.startswith('column'):
                return 0
        if table and portion.startswith('table'):
            return 0
    return 1


def normalize_sql(sql):
    if len(sql) > 0 and sql[-1] == ';':
        sql = sql[:-1]
    sql = " ".join(lower(sql).strip().split())
    sql = sql.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    for kword in ["count", "avg", "sum", "min", "max"]:
        sql = sql.replace(kword + " (", kword + "(")
    return sql.replace(") ,", "),")


def lower(s):
    s = s.replace("``", "`")
    lowers = ""
    current_quote = None
    for c in s:
        if current_quote is None:
            lowers += c.lower()
            if c in ['"', '\'', '`']:
                current_quote = c
        else:
            lowers += c
            if c == current_quote:
                current_quote = None
    return lowers


def generate_logical_beam(question, template_model, template_tokenizer, fill_model, fill_tokenizer, db_id,
                          schema_without_content, beam_width=5, num_outputs=5):
    templates = generate_logical_template(question, template_model, template_tokenizer, db_id, schema_without_content,
                                          beam_width)
    sqls = template_fill(question, fill_model, fill_tokenizer, db_id, schema_without_content, templates, beam_width,
                         num_outputs)
    return sqls


def generate_sql(db_id, question, schema_with_content, schema_without_content, llm_model, mode, generator=None,
                 t2s_model=None,
                 t2s_tokenizer=None, template_model=None, template_tokenizer=None, fill_model=None,
                 fill_tokenizer=None):
    if mode == "prompt":
        prompt = (PROMPT_DETECTION
                  .replace('[QUESTION]', question)
                  .replace('[SCHEMA_WITH_CONTENT]', schema_with_content))
    else:
        prompt = (PROMPT_BASE
                  .replace('[QUESTION]', question)
                  .replace('[SCHEMA_WITH_CONTENT]', schema_with_content))
    sql = ""
    flag = 0
    while not sql or sql.lower().split()[0] not in ["select", "with"]:
        if flag == 1:
            print(sql)
        if "gpt" in llm_model.lower():
            res = ask_llm(llm_model, prompt)
        elif "llama" in llm_model.lower():
            res = generate_vllm(llm_model, generator, prompt, mode)
        elif "t5" in llm_model.lower():
            if mode == "beam":
                res = generate_t5(question, t2s_model, t2s_tokenizer, db_id, schema_without_content, num_outputs=5)
            elif mode == "logical-beam":
                res = generate_logical_beam(question, template_model, template_tokenizer, fill_model, fill_tokenizer,
                                            db_id, schema_without_content, num_outputs=5)
            else:
                res = generate_t5(question, t2s_model, t2s_tokenizer, db_id, schema_without_content, num_outputs=1)
        else:
            break
        if isinstance(res, list):
            sql = res
        else:
            sql = parse_result(res)
        flag = 1
    return sql


def parse_result(res):
    res = res.replace("\n", " ")
    sql = ""
    if "```sql" in res:
        try:
            sql = re.findall("```sql(.*?)```", res)[0]
            sql = " ".join(sql.split()).strip()
        except:
            print(res)
    elif res.strip().lower().startswith("select"):
        sql = res.strip()
    return sql


def run_generation(input_path, output_path, llm_model, mode="base", generator=None, t5_checkpt_path=None,
                   template_gen_path=None, template_fill_path=None):
    t2s_model = None
    t2s_tokenizer = None
    template_model = None
    template_tokenizer = None
    fill_model = None
    fill_tokenizer = None
    if "t5" in llm_model.lower():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        t2s_model = load_t5_model(device=device, checkpt_dir=t5_checkpt_path)
        t2s_tokenizer = load_t5_tokenizer(checkpt_dir=t5_checkpt_path)
    if llm_model == "logical-beam":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        template_model = load_t5_model(device=device, checkpt_dir=template_gen_path)
        template_tokenizer = load_t5_tokenizer(checkpt_dir=template_gen_path)
        fill_model = load_t5_tokenizer(checkpt_dir=template_fill_path)
        fill_tokenizer = load_t5_tokenizer(checkpt_dir=template_fill_path)
    dataset = load_json(input_path)
    idx = 0
    out = []
    if os.path.exists(output_path):
        out = load_json(output_path)
        idx = len(out)
    for i, data in enumerate(tqdm(dataset)):
        if i < idx:
            continue
        schema_with_content = data["schema_with_content"]
        schema_without_content = data["schema_without_content"]
        question = data["question"]
        if mode == "clear":
            question = data["dq"]
            schema_with_content = data["schema_with_content_dq"]
            schema_without_content = data["schema_without_content_dq"]
        db_id = data["db_id"]
        sql = generate_sql(db_id, question, schema_with_content, schema_without_content, llm_model, mode, generator,
                           t2s_model,
                           t2s_tokenizer, template_model, template_tokenizer, fill_model, fill_tokenizer)
        if mode in ["base", "clear"]:
            data["predict_sql"] = sql
        else:
            data["predict_sqls"] = sql if isinstance(sql, list) else [sql]
        out.append(data)
        write_to_json(out, output_path)


def run_one_example():
    llm_model = "gpt-4o"
    mode = "base"

    dataset = load_json("./clambsql.json")
    data = dataset[0]
    schema_with_content = data["schema_with_content"]
    schema_without_content = data["schema_without_content"]
    question = "Show me feature articles of 1,000 words and news articles."
    db_id = data["db_id"]
    sql = generate_sql(db_id, question, schema_with_content, schema_without_content, llm_model, mode)
    print(sql)


def get_all_columns(gold_sql):
    try:
        sql_parsed = sql_metadata.Parser(gold_sql)
        tokens = sql_parsed.tokens
        sql_tokens = []
        for token in sql_parsed.non_empty_tokens:
            if token.value.startswith("`"):
                column = token.value.strip("`")
                idx = gold_sql.index(column)
                if gold_sql[idx - 2] != ".":
                    sql_tokens.append(column)
        columns_parsed = list(sql_parsed.columns)
        tables_parsed = sql_parsed.tables
        columns_all = []
        for column in columns_parsed + sql_tokens:
            if "." not in column:
                column = f"{tables_parsed[0]}.{column}"
            if column not in columns_all:
                columns_all.append(column)
    except:
        columns_all = []
    return columns_all

def llm_selection(question, token, p, sqls):
    sqls_options = [f"[{i+1}] {sql}" for i, sql in enumerate(sqls)]
    sqls_options.append(f"[{len(sqls)+1}] Neither of above")
    sql_options_str = "\n".join(sqls_options)
    prompt = ('''## Clarification task for Text-to-SQL
### Given the question with some corresponding SQLs, choose one SQL that represents the interpretation for the ambiguous context in the question.
### Return the option number of SQL without any explanation.

Question: [QUESTION]
Ambiguous Context: [CONTEXT]
Interpretation: [INTERPRETATION]
SQLs: 
[SQLS]

Answer: '''
              .replace("[QUESTION]", question)
              .replace("[CONTEXT]", token)
              .replace("[INTERPRETATION]", p)
              .replace("[SQLS]", sql_options_str))
    res_list = ask_llm("gpt-4o", prompt, n=10)
    res = get_best_result(res_list)
    try:
        idx = int(res)
    except:
        try:
            idx = int(re.findall("\[(.*?)]", res)[0])
        except:
            idx = random.randint(0, len(sqls) - 1)
    idx = idx - 1
    if idx == len(sqls):
        return ""
    else:
        return sqls[idx]

def clarification_multi_sql(result_path, base_path):
    result = load_json(result_path)
    base = load_json(base_path)
    for data in tqdm(result):
        predict_sqls = data["predict_sqls"]
        clear_ambiguity = parse_ambiguity_map(data["clear_ambiguity"])
        pred_sql = ""

        for token in clear_ambiguity:
            p = clear_ambiguity[token]
            if isinstance(p, str):  # query
                predict_sqls.append("")
                pred_sql = random.choice(predict_sqls)
            else:  # match
                for table in p:
                    for col in p[table]:
                        for sql in predict_sqls:
                            if table.lower() in sql.lower() and col.lower() in sql.lower():
                                pred_sql = sql
                                break
        if not pred_sql:
            pred_sql = base[data["index"]]["predict_sql"]
        data["predict_sql"] = pred_sql
    write_to_json(result, result_path)



if __name__ == '__main__':
    db_root_path = "./database"
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model", type=str)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    llm_model = args.llm_model
    mode = args.mode
    api_url = ""
    t5_checkpt_path = ""
    template_gen_path = ""
    template_fill_path = ""
    in_path = "../data/clambsql/clear/result_rewriting.json"
    model_name = llm_model.split('/')[-1].replace('-', '_')
    result_path = f"../data/clambsql/prediction/result_{model_name}_{mode}.json"
    generator = OpenAI(base_url=api_url, api_key="xx")
    run_generation(in_path, result_path, llm_model, mode)

