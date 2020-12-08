import json
import os
import random


def json_to_path(orig_json_path='/core/data/dataset/biaozhu/yf_zkzh/zkzh_sel_negative.json'):
    with open(orig_json_path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        print('{}: {}images {}knifes'.format(os.path.split(orig_json_path)[1], len(json_data['images']),
                                             len(json_data['annotations'])))
    recog_data = []
    random.shuffle(json_data['images'])
    for image in json_data['images']:
        data = image['file_name'] + '\n'
        recog_data.append(data)

    with open('/core/data/dataset/biaozhu/yf_select/zkzh_sel_negative.txt', 'w') as f:
        f.writelines(recog_data)
    return


if __name__ == "__main__":
    json_to_path()
