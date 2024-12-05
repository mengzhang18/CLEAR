import json
import re

def interpretations_to_context(column_interpretations):
    schemas = {}
    for column_interpretation in column_interpretations:
        tmp = re.findall("(.*): In table \"(.*)\"\. (.*)", column_interpretation)[0]
        column_name = tmp[0]
        table_name = tmp[1]
        interpretation = tmp[2]
        if table_name in schemas:
            schemas[table_name] += f" {column_name} ({interpretation}) ,"
        else:
            schemas[table_name] = f"{table_name} : {column_name} ({interpretation}) ,"
    prompt_context = ""
    for i, table in enumerate(schemas.values()):
        prompt_context += f'[{i + 1}] {table.rstrip(" ,").strip()}\n'
    return prompt_context

COLUMN_SHOT = '''Schema:
[1] department : creation (The year the department was created, like "1989", "1977", and "1989".) , title (The name of the department, like "Commerce", "Justice", and "Interior".) , department_id (The unique identifier of the department, like 11, 4, and 11.)
[2] head : born_state (The state where the head of the department was born, like "California", "Delaware", and "Alabama".) , head_id (The unique identifier of the head, like 2, 7, and 6.) , age (The age of the head of the department, like 69.0, 56.0, and 69.0.) , name (The name of the head of the department, like "Stewart Cink", "Billy Mayfair", and "Peadraig Harrington".) , head_name (The name of the head of the department, like "Stewart Cink", "Billy Mayfair", and "Peadraig Harrington".)

Question: List the name, born state and age of the heads of departments ordered by age.

Note: NULL

Disambiguations:
Let's think step by step. According to the schema with column interpretations, the "title" and "head_name" columns in the "head" table both record the name of the head, which is required for the question. Besides, the note is NULL. Therefore, there exists an ambiguity between "title" and "head_name" column for this question.
Ambiguity Mapping: ```json {"name": [{"head": ["title"]}, {"head": ["head_name"]}]} ```'''

TABLE_SHOT = '''Schema:
[1] department : creation (The year the department was created, like "1989", "1977", and "1989".) , name (The name of the department, like "Commerce", "Justice", and "Interior".) , department_id (The unique identifier of the department, like 11, 4, and 11.)
[2] admin : born_state (The state where the head of the department was born, like "California", "Delaware", and "Alabama".) , head_id (The unique identifier of the head, like 2, 7, and 6.) , age (The age of the head of the department, like 69.0, 56.0, and 69.0.) , name (The name of the head of the department, like "Stewart Cink", "Billy Mayfair", and "Peadraig Harrington".)
[3] supervisor : born_state (The state where the head of the department was born, like "California", "Delaware", and "Alabama".) , head_id (The unique identifier of the head, like 2, 7, and 6.) , age (The age of the head of the department, like 69.0, 56.0, and 69.0.) , name (The name of the head of the department, like "Stewart Cink", "Billy Mayfair", and "Peadraig Harrington".)

Question: List the name, born state and age of the heads of departments ordered by age.

Note: NULL

Disambiguations:
Let's think step by step. According to the schema with column interpretations, the name, born state and age of the heads mentioned in the question can be found in both "admin" and "supervisor" table with "name", "born_state" and "age" columns. Besides, the note is NULL. Therefore, there exists an ambiguity between "admin" and "supervisor" table.
Ambiguity Mapping: ```json {"name": [{"admin": ["name"]}, {"supervisor": ["name"]}], "age": [{"admin": ["age"]}, {"supervisor": ["age"]}], "born state": [{"admin": ["born_state"]}, {"supervisor": ["born_state"]}]} ```
'''

JOIN_SHOT = '''Schema:
[1] department : creation (The year the department was created, like "1989", "1977", and "1989".) , name (The name of the department, like "Commerce", "Justice", and "Interior".) , department_id (The unique identifier of the department, like 11, 4, and 11.)
[2] head : born_state (The state where the head of the department was born, like "California", "Delaware", and "Alabama".) , head_id (The unique identifier of the head, like 2, 7, and 6.) , age (The age of the head of the department, like 69.0, 56.0, and 69.0.) , name (The name of the head of the department, like "Stewart Cink", "Billy Mayfair", and "Peadraig Harrington".)
[3] head_born : head_id (The unique identifier of the head, like 1, 2, and 3.) , born_state (The state where the head of the department was born, like "California", "Delaware", and "Alabama".)

Question: List the name, born state and age of the heads of departments ordered by age.

Note: NULL

Disambiguations:
Let's think step by step. According to the schema with column interpretations, the born state mentioned in the question are recorded in both "head" and "head_born" table. Besides, the note is NULL.Therefore, there exists an ambiguity between "head" and "head_born" table.
Ambiguity Mapping: ```json {"born state": [{"head": ["born_state"]}, {"head_born": ["born_state"]}]} ```
'''

AGGREGATE_SHOT = '''Schema:
[1] department : creation (The year that the department was created, like "1789", "1903", and "1977".) , budget (The budget of the department, like 46.2, 10.7, and 23.4.) , department_id (The unique identifier of the department, like 10, 10, and 9. It may be the primary key of table "department".) , name (The name of the department, like "Interior", "Treasury", and "Labor".)
[2] department_budget : name (The name of the department, like "Agriculture", "Treasury", and "Justice".) , department_id (The unique identifier of the department, like 8, 10, and 12. It may be a foreign key linking to the department table.) , max_budget (The maximum budget allocated to the department, like 21.5, 23.4, and 44.6.) , avg_budget (The average budget allocated to the department, like 21.5, 23.4, and 44.6.) , total_budget(The total budget allocated to the department, like 21.5, 23.4, and 44.6.)

Question: What are the maximum and average budget of the departments?

Note: NULL

Disambiguations:
Let's think step by step. The budget in table "department" can be aggregated to compute the maximum budget, and max_budget directly record the max budget, so they both can be used to answer the question. Therefore there is ambiguity in between budget and max_budget. The budget in table "department" can be aggregated to compute the average budget, and avg_budget directly record the average budget, so they both can be used to answer the question. Besides, the note is NULL. Therefore there is ambiguity in between budget and avg_budget.
Ambiguity Mapping: ```json {"maximum": [{"department": ["budget"]}, {"department_budget": ["max_budget"]}], "average": [{"department": ["budget"]}, {"department_budget": ["avg_budget"]}]} ```
'''

PROMPT_BASE = '''### You are an expert in writing SQL. To answer the question, you need to find some relevant tables and columns based on the schema below whose format like {table_name: column_name (interpretation of column)}. 
### Ambiguity is that the entity in the question can correspond to multiple possible columns involving similar information. 
### Find the ambiguous columns. Output the Ambiguity Mapping, a two-layer JSON structure, where the key of the first layer is entity in the question, which corresponds to the second layer, an array contains elements mapping a table name to an array of column names.
### Please consider the note if it is not NULL.

---
Here are examples.

[FEW_SHOTS]

---

Schema:
[SCHEMA_WITH_INTERPRETATION]

Question: [QUESTION]

Note: [NOTE]

Disambiguations:'''

def get_prompt_ambiguity_map(question, column_interpretations, note="NULL"):
    prompt_context = interpretations_to_context(column_interpretations)
    prompt = (PROMPT_BASE
              .replace("[FEW_SHOTS]", "\n\n".join([COLUMN_SHOT, TABLE_SHOT, JOIN_SHOT, AGGREGATE_SHOT]))
              .replace("[SCHEMA_WITH_INTERPRETATION]", prompt_context)
              .replace("[QUESTION]", question)
              .replace("[NOTE]", note))
    return prompt

def get_prompt_ambiguity_map_wo_enrichment(question, column_interpretations):
    schemas = {}
    for column_interpretation in column_interpretations:
        tmp = column_interpretation.split(".")
        column_name = tmp[1]
        table_name = tmp[0]
        if table_name in schemas:
            schemas[table_name] += f" {column_name} ,"
        else:
            schemas[table_name] = f"{table_name} : {column_name} ,"
    prompt_context = ""
    for i, table in enumerate(schemas.values()):
        prompt_context += f'[{i + 1}] {table.rstrip(" ,").strip()}\n'
    prompt = ""
    return prompt

def get_prompt_ambiguity_map_column(question, column_interpretations, note="NULL"):
    prompt_context = interpretations_to_context(column_interpretations)
    prompt = (PROMPT_BASE
              .replace("[FEW_SHOTS]", COLUMN_SHOT)
              .replace("[SCHEMA_WITH_INTERPRETATION]", prompt_context)
              .replace("[QUESTION]", question)
              .replace("[NOTE]", note))
    return prompt

def get_prompt_ambiguity_map_table(question, column_interpretations, note="NULL"):
    prompt_context = interpretations_to_context(column_interpretations)
    prompt = (PROMPT_BASE
              .replace("[FEW_SHOTS]", TABLE_SHOT)
              .replace("[SCHEMA_WITH_INTERPRETATION]", prompt_context)
              .replace("[QUESTION]", question)
              .replace("[NOTE]", note))
    return prompt

def get_prompt_ambiguity_map_join(question, column_interpretations, note="NULL"):
    prompt_context = interpretations_to_context(column_interpretations)
    prompt = (PROMPT_BASE
              .replace("[FEW_SHOTS]", JOIN_SHOT)
              .replace("[SCHEMA_WITH_INTERPRETATION]", prompt_context)
              .replace("[QUESTION]", question)
              .replace("[NOTE]", note))
    return prompt

def get_prompt_ambiguity_map_aggregate(question, column_interpretations, note="NULL"):
    prompt_context = interpretations_to_context(column_interpretations)
    prompt = (PROMPT_BASE
              .replace("[FEW_SHOTS]", AGGREGATE_SHOT)
              .replace("[SCHEMA_WITH_INTERPRETATION]", prompt_context)
              .replace("[QUESTION]", question)
              .replace("[NOTE]", note))
    return prompt

def get_prompt_column_interpretation(schema_with_content):
    tables = schema_with_content.split("|")
    schema = ""
    for i, table in enumerate(tables):
        table_name = table.split(":")[0].strip()
        column_with_content = table[table.index(":") + 1:].strip()
        schema += f"[{i + 1}] {table_name}\n{column_with_content}\n"
    prompt = f'''You are a database expert. I will give you a table containing the column names and the first three rows of content for each column. Please infer the meaning of each column based on the table names, column names, and content. Give the explanation and detailed information could be inferred for the column.
    Directly output the explanations of columns like the examples.

    ---
    Here are some examples.

    Database schema:
    [1] singer
    singer_id (1, 6, 4), name ("Joe Sharp", "Tribal King", "Timbaland"), song_name ("Sun", "You", "Gentleman"), song_release_year ("2014", "2008", "1992"), age (41, 25, 29)

    Explanation of column:
    [1] singer
    singer_id : In table "singer". The unique identifier of singer, like 1, 6 and 4. It may be the primary key of table "singer".
    name : In table "singer". The name of singer, like "Joe Sharp", "Tribal King", and "Timbaland". 
    song_name : In table "singer". the name of song, like "Sun", "You", and "Gentleman".
    song_release_year : In table "singer". the year that the song is released, like "2014", "2008", and "1992".
    ---

    Database schema:
    {schema}

    Explanation of column:'''

    return prompt

def get_prompt_query_ambiguity(question):
    prompt = f'''## Given a query with scope ambiguity or attachment ambiguity, identify them with a ambiguity mapping in a JSON format, which records all of the possible interpretations.
    ## If there is no ambiguity, output the empty map.

    Query: Show me the camera each man observes.
    Ambiguity mapping: ```json {{"each man": ["for each man individually", "common to all men"]}} ```

    Query: What is the brand of the hat that every girl lifts?
    Ambiguity mapping: ```json {{"every girl": ["for every girl individually", "common to all girls"]}} ```

    Query: List the dog that each woman feds?
    Ambiguity mapping: ```json {{"each woman": ["for every woman individually", "common to all women"]}} ```

    Query: What is the kidnapper that kidnaps the fat girl and boy?
    Ambiguity mapping: ```json {{"fat": ["girl", "girl and boy"]}} ```

    Query: List a girl and a boy crying.
    Ambiguity mapping: ```json {{"crying": ["a boy", "a girl and a boy"]}} ```

    Query: Give the cats and the dogs that chase the blue fish and bird.
    Ambiguity mapping: ```json {{"that chase the blue fish and bird": ["the dogs", "the cats and the dogs"]}} ```

    Query: {question}
    Ambiguity mapping: '''
    return prompt