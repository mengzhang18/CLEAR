import os
import time

import openai

from dotenv import load_dotenv

load_dotenv()

def ask_llm(model, prompt, temperature=0.0, max_tokens=2048, stop=None, n=1, api_base=None, api_key=None):
    # print(prompt)
    llm = openai.OpenAI()
    if api_base and api_key:
        llm = openai.OpenAI(base_url=api_base, api_key=api_key)

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


if __name__ == '__main__':
    prompt = '''hello'''
    print(ask_llm("gpt-4o", prompt))



