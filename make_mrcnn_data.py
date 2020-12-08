import json
import os
import random
import time


def create_data(orig_json_path=''):
    """
    生成 positive 和 negative 数据集
    """
    with open(orig_json_path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    print("%-30s: [%-6d] images  [%-6d] knifes  note: %s" % (
        orig_json_path.split('/', 5)[-1], len(json_data['images']), len(json_data['annotations']), 'all_office_data'))
    # 第一次处理：过滤掉刀特别多的图片
    del_set1 = set()
    dct = {}
    for image in json_data['images']:
        if image['id'] not in dct:
            dct[image['id']] = 0
    for annotation in json_data['annotations']:
        dct[annotation['image_id']] += 1
    dct_val = sorted(dct.items(), key=lambda x: x[1], reverse=True)
    for val in dct_val:
        if val[1] >= 8:
            del_set1.add(val[0])
        else:
            break
    print("One-step of del-image: ", len(del_set1))
    print(del_set1)
    # 第二次处理：过滤图片尺寸过小的图片
    del_set2 = set()
    # for image in json_data['images']:
    #     if image['height'] < 200 or image['width'] < 200:
    #         del_set2.add(image['id'])
    # print("Two-step of del-image: ", len(del_set2))
    # print(del_set2)
    del_set = del_set1 | del_set2
    print("All of del-image: ", len(del_set))
    print(del_set)
    # 写入新文件
    # # new_json_path = os.path.split(orig_json_path)[0] + '/new_' + os.path.split(orig_json_path)[1][9:]
    # new_json_path = os.path.split(orig_json_path)[0] + '/new_' + os.path.split(orig_json_path)[1]
    # new_json_data = {'images': [], 'annotations': [],
    #                  'categories': [{"supercategory": "none", "id": 1, "name": "knife"}]}
    # for image in json_data['images']:
    #     if image['id'] not in del_set:
    #         new_json_data['images'].append(image)
    # for annotation in json_data['annotations']:
    #     if annotation['image_id'] not in del_set:
    #         new_json_data['annotations'].append(annotation)
    # print("%-30s: [%-6d] images  [%-6d] knifes  note: %s" % (
    #     new_json_path.split('/', 5)[-1], len(new_json_data['images']), len(new_json_data['annotations']),
    #     'del >=4 <300'))
    # with open(new_json_path, 'w', encoding='utf8')as fp:
    #     json.dump(new_json_data, fp)
    #     print("写入 %s 成功！" % new_json_path)
    return


def split_data(val_rate=1):
    """
    随机划分数据集
    """
    new_json_path = os.path.split(orig_json_path)[0] + '/' + os.path.split(orig_json_path)[1][9:]
    with open(new_json_path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    num_images = len(json_data['images'])
    print("%-30s: [%-6d] images  [%-6d] knifes  note: %s" % (
        new_json_path.split('/', 5)[-1], num_images, len(json_data['annotations']),
        'val_rate = ' + str(val_rate)))
    # 如果val_rate为1，则自动划分比例
    if val_rate == 1:
        # 对于negative恒定为三七分
        if new_json_path[-14:] == '_negative.json':
            val_rate = 0.3
        elif num_images > 10000:
            val_rate = 0.1
        elif num_images <= 3000:
            val_rate = 0.3
        else:
            val_rate = 0.2
    val_index = random.sample(range(0, len(json_data['images'])), k=int(len(json_data['images']) * val_rate))
    val_id = []
    train_json_data = {'images': [], 'annotations': [],
                       'categories': [{"supercategory": "none", "id": 1, "name": "knife"}]}
    val_json_data = {'images': [], 'annotations': [],
                     'categories': [{"supercategory": "none", "id": 1, "name": "knife"}]}
    for index, image in enumerate(json_data['images']):
        if index in val_index:
            val_json_data['images'].append(image)
            val_id.append(image['id'])
        else:
            train_json_data['images'].append(image)
    for annotation in json_data['annotations']:
        if annotation['image_id'] in val_id:
            val_json_data['annotations'].append(annotation)
        else:
            train_json_data['annotations'].append(annotation)
    # 路径自动加train和val的前缀
    train_json_path = '/'.join(new_json_path.split('/')[:-1]) + '/train_' + new_json_path.split('/')[-1]
    val_json_path = '/'.join(new_json_path.split('/')[:-1]) + '/val_' + new_json_path.split('/')[-1]
    print("%-30s: [%-6d] images  [%-6d] knifes  note: %s" % (
        train_json_path.split('/', 5)[-1], len(train_json_data['images']), len(train_json_data['annotations']),
        'val_rate = ' + str(val_rate)))
    print("%-30s: [%-6d] images  [%-6d] knifes  note: %s" % (
        val_json_path.split('/', 5)[-1], len(val_json_data['images']), len(val_json_data['annotations']),
        'val_rate = ' + str(val_rate)))
    with open(train_json_path, 'w', encoding='utf8')as fp:
        json.dump(train_json_data, fp)
        print("写入 %s 成功！" % train_json_path)
    with open(val_json_path, 'w', encoding='utf8')as fp:
        json.dump(val_json_data, fp)
        print("写入 %s 成功！" % val_json_path)
    return


def update_dataset_json(json_dataset_path='/data4t/data/zhangyufei/data/dataset.json', log_name='dataset',
                        dataset_root='/core/data/dataset/biaozhu'):
    json_dataset = {}
    txt_dataset_path = os.path.join(os.path.split(json_dataset_path)[0], log_name)

    # 需要添加的子目录,并按字母排序
    dataset_dirs = []
    for root, dirs, files in os.walk(dataset_root):
        for dataset_dir in dirs:
            if dataset_dir[:3] == 'yf_':
                dataset_dirs.append(os.path.join(root, dataset_dir))
    dataset_dirs.sort()

    for dataset_dir in dataset_dirs:

        # 去除没有json的文件夹，因为有些文件夹存了很多jpg
        flag = True
        for file_name in os.listdir(dataset_dir):
            if flag and os.path.splitext(file_name)[1] == '.json':
                flag = False
        if flag: continue

        dataset_name = os.path.basename(dataset_dir)[3:]  # 将数据的分类标签存起来，[3:]不显示yf_
        json_dataset[dataset_name] = []

        # 将文件名+时间戳存入files_list并排序
        files_list = []
        for file_name in os.listdir(dataset_dir):
            if os.path.splitext(file_name)[1] == '.json':
                time_stamp = os.path.getmtime(os.path.join(dataset_dir, file_name))
                files_list.append([time_stamp, file_name])
        files_list.sort()

        for index, [time_stamp, file_name] in enumerate(files_list):
            time_str = time.strftime('%m/%d/%H:%M', time.localtime(time_stamp))
            file_path = os.path.join(dataset_dir, file_name)
            file_no_suffix = os.path.splitext(file_name)[0]  # 分离文件名字
            with open(file_path, 'r', encoding='utf8')as fp:
                json_data = json.load(fp)
            json_dataset[dataset_name].append(
                {'id': index + 1, 'file_name': file_no_suffix, 'file_path': file_path, 'modified_time': time_str,
                 'images': len(json_data['images']), 'annotations': len(json_data['annotations']), 'note': ''})

    with open(json_dataset_path, 'w', encoding='utf8')as fp:
        json.dump(json_dataset, fp, indent=4)
        print("更新 {} 成功！".format(json_dataset_path))

    fp = open(txt_dataset_path, 'w')
    for dataset_name in json_dataset:
        print('%s < %s > %s' % ('-' * 40, dataset_name, '-' * 40,), file=fp)
        for item in json_dataset[dataset_name]:
            print('\'%s\': \'%s\',' % (item['file_name'], item['file_path']), file=fp)
    print('', file=fp)
    for dataset_name in json_dataset:
        print('%s < %s > %s' % ('-' * 40, dataset_name, '-' * 40,), file=fp)
        for item in json_dataset[dataset_name]:
            print('%s [%s][%6d/%6d] %-20s' % (
                chr(item['id'] + 96), item['modified_time'], item['images'], item['annotations'], item['file_name']),
                  file=fp)
    fp.close()
    print("写入 {} 成功！".format(txt_dataset_path))


def sample_neg():
    """
    随机采样数据集
    """
    sh, wx = [], []
    print("shanghai_neg is %d, wuxi_neg is %d: " % (len(sh), len(wx)))

    orig_json_path = "/core/data/dataset/biaozhu/yf_wuxi/orig_wuxi_negative.json"
    with open(orig_json_path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    print("The num of images: ", len(json_data['images']))
    print("The num of knifes: ", len(json_data['annotations']))
    train_id = []
    train_json_data = {'images': [], 'annotations': [],
                       'categories': [{"supercategory": "none", "id": 1, "name": "knife"}]}
    val_json_data = {'images': [], 'annotations': [],
                     'categories': [{"supercategory": "none", "id": 1, "name": "knife"}]}

    for image in json_data['images']:
        if image['id'] in wx:
            train_json_data['images'].append(image)
        else:
            val_json_data['images'].append(image)

    print("The num of images: ", len(train_json_data['images']))
    print("The num of knifes: ", len(train_json_data['annotations']))
    print("The num of images: ", len(val_json_data['images']))
    print("The num of knifes: ", len(val_json_data['annotations']))

    train_json_path = '/core/data/dataset/biaozhu/yf_wuxi/train_wuxi_negative.json'
    with open(train_json_path, 'w', encoding='utf8')as fp:
        json.dump(train_json_data, fp)
        print("写入 %s 成功！" % train_json_path)
    val_json_path = '/core/data/dataset/biaozhu/yf_wuxi/val_wuxi_negative.json'
    with open(val_json_path, 'w', encoding='utf8')as fp:
        json.dump(val_json_data, fp)
        print("写入 %s 成功！" % val_json_path)
    return


def merge_neg(orig_json_path='/core/data/dataset/biaozhu/yf_new_test/zkzh689.json'):
    with open(orig_json_path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        print('{}: {}images {}knifes'.format(os.path.split(orig_json_path)[1], len(json_data['images']),
                                             len(json_data['annotations'])))

    neg_json_path = orig_json_path.replace('.json', '_negative.json')
    # neg_json_path = orig_json_path.replace('.json', '_invalid.json')
    with open(neg_json_path, 'r', encoding='utf8')as fp:
        json_data_neg = json.load(fp)
        print('{}: {}images {}knifes'.format(os.path.split(neg_json_path)[1], len(json_data_neg['images']),
                                             len(json_data_neg['annotations'])))

    for image in json_data_neg['images']:
        json_data['images'].append(image)
    new_json_path = os.path.split(orig_json_path)[0] + '/mg_' + os.path.split(orig_json_path)[1]
    with open(new_json_path, 'w', encoding='utf8')as fp:
        json.dump(json_data, fp)
        print('{}: {}images {}knifes'.format(os.path.split(new_json_path)[1], len(json_data['images']),
                                             len(json_data['annotations'])))
    return


if __name__ == "__main__":
    merge_neg()
    update_dataset_json()
