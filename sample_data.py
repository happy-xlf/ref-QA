#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :sample_data.py
# @Time      :2024/09/25 14:41:17
# @Author    :Lifeng
# @Description :
import json
import os
import random

def read_dir_file(dir_path):
    path_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".jsonl"):
                path_list.append(os.path.join(root, file))
    return path_list


def read_json_file(path):
    res = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            res.append(data)
    return res


if __name__ == "__main__":
    dir_path = "/home/yuanyuan06/xulifeng/Rag_work/code/output"
    path_list = read_dir_file(dir_path)
    res = []
    out_file = "./rag_qa_test.jsonl"
    for path in path_list:
        data = read_json_file(path)
        if len(data) <100:
            for it in data:
                if "参考内容" not in it["answer"]:
                    res.append(it)
        else:
            tmp = []
            new_data = random.sample(data, 200)
            for it in new_data:
                if "参考内容" not in it["answer"]:
                    tmp.append(it)
                    if len(tmp) == 100:
                        break
            res.extend(tmp)
    print(len(res))
    with open(out_file, "w", encoding="utf-8") as f:
        for it in res:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
