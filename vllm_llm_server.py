#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :llm_fastapi.py
# @Time      :2024/07/19 11:59:39
# @Author    :Lifeng
# @Description :
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from vllm import LLM, SamplingParams

model_name_or_path = "/home/yuanyuan06/model/qwen2-72b-instruct"

llm = LLM(
    tokenizer=model_name_or_path,
    model=model_name_or_path,
    tensor_parallel_size=8,
    trust_remote_code=True,
    max_model_len=8096,
    gpu_memory_utilization=0.8
)

tokenizer = llm.get_tokenizer()

# 创建FastAPI应用实例
app = FastAPI()


# 定义请求体模型，与OpenAI API兼容
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 1024
    temperature: float = 1.0


# 文本生成函数
def generate_text(messages: list, max_tokens: int, temperature: float):

    text = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # max_token to control the maximum output length
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.8,
        repetition_penalty=1.2,
        max_tokens=max_tokens)

    outputs = llm.generate([text], sampling_params)
    response = outputs[0].outputs[0].text
    return response


# 定义路由和处理函数，与OpenAI API兼容
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # 调用自定义的文本生成函数
    response = generate_text(
        request.messages, request.max_tokens, request.temperature
    )
    return {"choices": [{"message": {"role": "assistant", "content": response}}], "model": request.model}

# 启动FastAPI应用
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)