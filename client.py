from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
import time

os.environ["OPENAI_API_KEY"] = "None"

# def llm_chat(model_name) -> ChatOpenAI:
#     llm = ChatOpenAI(model_name=model_name, base_url= "http://localhost:9999/v1")
#     return llm

def vllm_chat(model_name) -> ChatOpenAI:
    llm = ChatOpenAI(model_name=model_name, base_url= "http://localhost:7000/v1")
    return llm

if __name__ == "__main__":
    vllm = vllm_chat(model_name= "Qwen2-72B")
    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content="写一个快速排序代码"),
    ]

    ai_message = vllm.invoke(
        messages,
        temperature=0.2,  # 设置温度
        max_tokens=1024,  # 设置最大 token 长度
    )
    print(ai_message.content)
