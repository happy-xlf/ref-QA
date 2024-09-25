import os
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
import json
from tqdm import tqdm

# 创建log日志器
import logging
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='test.log',
                    filemode='w')


def vllm_chat(model_name) -> ChatOpenAI:
    llm = ChatOpenAI(model_name=model_name, base_url= "http://localhost:7000/v1")
    return llm

def format_docs(docs):
    return "\n".join(
        [f"参考内容{i+1}\n{doc.page_content}\n" for i, doc in enumerate(docs)]
    )

os.environ["OPENAI_API_KEY"] = "None"
emb_model = "/home/yuanyuan06/xulifeng/Rag_work/model/bce-embedding-base_v1"
embedding_model_kwargs = {'device': 'cuda:0'}
embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
)

llm = vllm_chat(model_name= "Qwen2-72B")

template = """你是一名根据参考内容回答用户问题的机器人，你的职责是：根据提供的参考内容回答用户的问题。如果参考内容与问题不相关，你可以选择忽略参考内容，只回答问题。

##参考内容：
{context}

##用户问题：
{question}

请根据参考内容，回答用户问题，回答内容要简洁，说重点，不要做过多的解释，输出内容限制在200字符内。
"""

def chat_llm(query):
    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content=query),
    ]
    ai_message = llm.invoke(
        messages,
        temperature=0.1,  # 设置温度
        max_tokens=300,  # 设置最大 token 长度
    )
    return ai_message.content


def get_chunk_retrival(text, file_path):
    documents = [Document(
                    page_content=text,
                    metadata={"source": file_path},
                )]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)

    retriver = db.as_retriever(search_kwargs={"k": 3})

    return retriver

def get_ans(retriver, query_list):
    pred_list = []
    for query in tqdm(query_list, desc="Processing", total=len(query_list)):
        # print(query)
        refs = retriver.invoke(query)
        context = format_docs(refs)
        prompt = template.format(context=context, question=query)
        logging.info(prompt)
        logging.info("=====================================")
        pred = chat_llm(prompt)
        logging.info(pred)
        logging.info("=====================================")
        pred_list.append(pred)
    return pred_list

def get_ref_answer(file_path, query_list, out_path):
    documents = TextLoader(file_path).load()
    print(f"Loadded {file_path}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)

    retriver = db.as_retriever(search_kwargs={"k": 3})

    for query in tqdm(query_list, desc="Processing", total=len(query_list)):
        refs = retriver.invoke(query)
        context = format_docs(refs)
        prompt = template.format(context=context, question=query)
        # print(prompt)
        # print("=====================================")
        pred = chat_llm(prompt)
        # print(pred)
        # print("=====================================")

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"query": query, "pred": pred}, ensure_ascii=False) + "\n")

def get_all_file_name(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_list.append(file.split(".")[0])
    return file_list

if __name__ == "__main__":
    file_name_list = get_all_file_name("/home/yuanyuan06/xulifeng/Rag_work/industry_pretrain_data")
    print(len(file_name_list))
    for idx, file_name in enumerate(file_name_list):
        print(f"Processing {idx+1}/{len(file_name_list)}")
        file_path = f"/home/yuanyuan06/xulifeng/Rag_work/code/query_data2/{file_name}.jsonl"
        doc_path = f"/home/yuanyuan06/xulifeng/Rag_work/industry_pretrain_data/{file_name}.txt"
        out_path = f"/home/yuanyuan06/xulifeng/Rag_work/code/answer_data/{file_name}_pred.jsonl"
        query_list = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                query_list.append(data["query"])
        get_ref_answer(doc_path, query_list, out_path)
        break








