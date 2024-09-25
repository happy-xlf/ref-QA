import os
from prompt import Query_Instruction
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
import re
import time
import json
from tqdm import tqdm
from retrival_doc import get_chunk_retrival, get_ans
os.environ["OPENAI_API_KEY"] = "None"

def vllm_chat(model_name) -> ChatOpenAI:
    llm = ChatOpenAI(model_name=model_name, base_url= "http://localhost:7000/v1")
    return llm

def get_all_file(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def get_file_content(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(line)
    data = "".join(data)
    return data

def chat_doc(text, llm):
    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content=text),
    ]
    ai_message = llm.invoke(
        messages,
        temperature=0.8,  # 设置温度
        max_tokens=512,  # 设置最大 token 长度
    )
    # print(ai_message.content)
    return ai_message.content

if __name__ == '__main__':
    llm = vllm_chat(model_name= "Qwen2-72B")
    dir_path = '/home/yuanyuan06/xulifeng/Rag_work/industry_pretrain_data'
    file_list = get_all_file(dir_path)
    for file_path in tqdm(file_list, desc="query_gen", total=len(file_list)):
        print("file_path: ", file_path)
        file_content = get_file_content(file_path)
        chunk_list = []
        chun_size = 2000
        for i in range(0, len(file_content), chun_size):
            chunk_list.append(file_content[i:i+chun_size])
        
        print("chunk_list: ", len(chunk_list))
        for idx, chunk in enumerate(chunk_list):
            print(f"这是第{idx}个")
            query_list = []
            if len(chunk) < 300:
                continue
            query = Query_Instruction.replace("{text}", chunk)
            output = chat_doc(query, llm)
            #设计re正则，提取text中的query内容
            
            pattern = r'"query": "(.*?)"'
            matches = re.findall(pattern, output)
            try:
                for match in matches:
                    query_list.append(match)
            except:
                print("error: llm_output")
            
            retriver = get_chunk_retrival(chunk, file_path)
            pred_list = get_ans(retriver, query_list)

            file_name = file_path.split("/")[-1].split(".")[0]
            if len(query_list) > 0:
                with open(f"/home/yuanyuan06/xulifeng/Rag_work/code/output/{file_name}.jsonl", "a") as f:
                    for i in range(len(query_list)):
                        f.write(json.dumps({"query": query_list[i], "answer": pred_list[i]}, ensure_ascii=False) + "\n")
                
