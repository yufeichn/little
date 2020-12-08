import os
import json
import numpy as np

from make_mrcnn_data import update_dataset_json


def bash_catalog(json_dataset_path, log_name, model_str, inference='inference',
                 expi_root='/core7/data/zhangyufei/expi'):
    txt_dataset_path = os.path.join(expi_root, model_str, model_str)

    # 生成dt的dict
    dt_dict = {}
    inference_path = os.path.join(expi_root, model_str, inference)
    for inference_dir in sorted(os.listdir(inference_path)):
        dt_dict[inference_dir] = os.path.join(inference_path, inference_dir, 'bbox.json')
    # 生成gt的dict
    gt_dict = {}
    with open(json_dataset_path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    for dataset_name in json_data:
        for item in json_data[dataset_name]:
            gt_dict[item['file_name']] = [item['file_path'], item['images'], item['annotations']]

    fp = open(txt_dataset_path, 'a')
    fp.write("\n{}model: {}{}\n".format('-' * 50, model_str, '-' * 50))
    for inf_dir in dt_dict:
        info_str = "[{:>5d}/{:>5d}] {}\n".format(gt_dict[inf_dir][1], gt_dict[inf_dir][2], inf_dir)
        info_cmd = "python /core7/data/zhangyufei/project/simple_detection/tools/eval_detection.py --dt {} --gt " \
                   "{}\n".format(dt_dict[inf_dir], gt_dict[inf_dir][0])
        fp.write(info_str)
        fp.write(info_cmd)
        cmd = "LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 " \
              "/core7/data/zhangyufei/miniconda3/envs/mh/bin/python " \
              "/core7/data/zhangyufei/project/simple_detection/tools/eval_detection.py " \
              "--dt {} --gt {}".format(dt_dict[inf_dir], gt_dict[inf_dir][0])
        print(cmd)
        cmd_out = os.popen(cmd).read()
        # print(cmd_out)
        fp.write(cmd_out[cmd_out.index('| Thresh'):])
    fp.close()


def cfg(json_dataset_path, cfg_dic):
    with open(json_dataset_path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)

    for mode in cfg_dic:
        res = []
        for dataset_name in cfg_dic[mode]:
            # char to num (abc to 123)
            lst = [ord(x) - 96 for x in list(cfg_dic[mode][dataset_name])]
            tmp = [(json_data[dataset_name][index - 1]['file_name'], json_data[dataset_name][index - 1]['images']) for
                   index in lst]
            res += tmp
        print('  {}: {}'.format(mode, tuple([_[0] for _ in res])))
        if mode == 'TRAIN':
            ratio = (0.4, 1.0, 0.1, 1.0)
            num = tuple([_[1] for _ in res])
            n = int(np.dot(ratio, num)) // 400 * 100
            epoch = (12, 7, 2)
            steps = (epoch[0] * n, (epoch[0] + epoch[1]) * n, (epoch[0] + epoch[1] + epoch[2]) * n)
            print('  %s: %s' % ('TRAIN_SAMPLE_RATIO', ratio))
            print('  %s: %s' % ('# TRAIN_IMAGE_NUM', num))
            print('  %s: %s' % ('# TRAIN_EPOCH_NUM', n))
            print('  # EPOCH: ({}, {}, {})'.format(epoch[0], epoch[1], epoch[2]))
            print('  TIME CORE10: ', steps[-1] / 15560)
            print('\n  %s: %s' % ('STEPS', steps[:2]))
            print('  %s: %s' % ('MAX_ITER', steps[-1]))


if __name__ == "__main__":
    json_path = '/core7/data/zhangyufei/data/dataset.json'

    # update_dataset_json(log_name='zkzh11-1')
    bash_catalog(json_path, 'zkzh11', model_str='zkzh11', inference='inference')

    # cfg(json_path, {
    #     'TEST': {'wscd': 'gi', 'new_zkzh': 'k'},
    #     'TRAIN': {'office': 'e', 'wscd': 'de', 'new_zkzh': 'l'}
    # })
