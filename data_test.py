import json
import os
import random

orig_json_path = "/core/data/dataset/biaozhu/yf_office/one_original_office.json"

with open(orig_json_path, 'r', encoding='utf8')as fp:
    json_data = json.load(fp)
print("%-30s: [%-6d] images  [%-6d] knifes  note: %s" % (
    orig_json_path.split('/', 5)[-1], len(json_data['images']), len(json_data['annotations']), 'all_office_data'))
# 第一次处理：过滤掉刀特别多的图片
new_set = set()
dct = {}
for annotation in json_data['annotations']:
    new_set.add(annotation['id'])
print("new_set: ", len(new_set))
print(new_set)

orig_json_path = "/core/data/dataset/biaozhu/yf_wuxi/original_val_wuxi.json"

with open(orig_json_path, 'r', encoding='utf8')as fp:
    json_data = json.load(fp)
print("%-30s: [%-6d] images  [%-6d] knifes  note: %s" % (
    orig_json_path.split('/', 5)[-1], len(json_data['images']), len(json_data['annotations']), 'all_office_data'))
# 第一次处理：过滤掉刀特别多的图片
old_set = set()
dct = {}
for annotation in json_data['annotations']:
    old_set.add(annotation['id'])
print("old_set: ", len(old_set))
print(old_set)

more = new_set - old_set
print("more: ", len(more))
print(more)

miss = old_set - new_set
print("miss: ", len(miss))
print(miss)