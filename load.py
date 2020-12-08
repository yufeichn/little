import base64
import json
import os
import numpy as np
import cv2
from cvtools import cv_load_image


# file1 = "/core/data/dataset/biaozhu/yf_cut/wuxi_cut.json"
# with open(file1, 'r', encoding='utf8')as fp:
#     json_data = json.load(fp)
#
# pic_dir_root = "/core/data/dataset/biaozhu/yf_cut/wuxi_cut"
# if not os.path.exists(pic_dir_root):
#     os.makedirs(pic_dir_root)
#
# # cv2.imshow("img", img)
# # cv2.waitKey(0)
# count = 0
# for image in json_data['images']:
#     if image['height'] > 1500 or image['width'] > 1500:
#         pic_file_name = os.path.join(pic_dir_root, str(image['id']) + '.jpg')
#         print(pic_file_name)
#         url = image['file_name']
#         img = cv_load_image(url)
#         cv2.imwrite(pic_file_name, img)
#         count += 1
#
# print(count)
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def show_gt():
    file1 = "/core/data/dataset/biaozhu/yf_filtr/air_test.json"
    with open(file1, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    count = 0
    cv2.namedWindow("enhanced", 0)
    cv2.resizeWindow("enhanced", 1920, 1080)

    for image in json_data['images']:
        url = image['file_name']
        img = cv_load_image(url)
        count += 1
        print(count)
        imgStack = stackImages(0.5, ([img, ],))
        cv2.imshow("enhanced", img)
        cv2.waitKey(0)


file1 = "/core/data/dataset/biaozhu/yf_wuxi/val_wuxi.json"
with open(file1, 'r', encoding='utf8')as fp:
    json_data = json.load(fp)

pic_dir_root = "/core/data/dataset/biaozhu/yf_box"
if not os.path.exists(pic_dir_root):
    os.makedirs(pic_dir_root)

count = 0
for image in json_data['images']:
    pic_file_name = os.path.join(pic_dir_root, str(image['id']) + '.jpg')
    print(pic_file_name)
    url = image['file_name']
    img = cv_load_image(url)

    # cv2.imwrite(pic_file_name, img)
    count += 1

print(count)
