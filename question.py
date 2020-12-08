import os
import torch
import cv2
import numpy as np
from cvtools import cv_load_image


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


def getContours(img, imgCopy):
    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    approxPolys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000: continue
        print("area: %d" % area)
        # 画出轮廓
        # (载体,轮廓,-1,颜色,厚度)
        cv2.drawContours(imgCopy, cnt, -1, (255, 0, 0), 3)
        # # 计算轮廓的长度
        # peri = cv2.arcLength(cnt, True)
        # # 计算近似拐角点
        # approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        # # 创建边界框
        # x, y, w, h = cv2.boundingRect(approx)
        # cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # # 创建近似矩形
        # approxPoly = cv2.approxPolyDP(cnt, 5, True)
        # cv2.polylines(imgCopy, [approxPoly], True, (0, 255, 0), 3)
        # approxPolys.append(approxPoly)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        # mask = cv2.GaussianBlur(mask, (5, 5), 0)
        cv2.imshow('mask', mask)

        # hull = cv2.convexHull(cnt)
        # print(hull)
        # length = len(hull)
        # for i in range(length):
        #     cv2.line(imgCopy, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 0, 255), 9)

        # hull = cv2.convexHull(cnt, returnPoints=False)
        # defects = cv2.convexityDefects(cnt, hull)
        # for i in range(defects.shape[0]):
        #     s, e, f, d = defects[i, 0]
        # start = tuple(cnt[s][0])
        # end = tuple(cnt[e][0])
        # far = tuple(cnt[f][0])
        # cv2.line(imgCopy, start, end, [0, 255, 0], 20)
        # cv2.circle(imgCopy, far, 10, [0, 0, 255], -1)
    return approxPolys


if __name__ == "__main__":
    root = "/home/zhangyufei/project/split"
    # cv2.namedWindow('test', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)

    # for name in os.listdir(root):
    #     filename = os.path.join(root, name)
    #     print(filename)
    #     img = cv2.imread(filename)
    #     imgContour = img.copy()
    #     # imgContour1 = img.copy()
    #     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    #     # imgBlur = cv2.GaussianBlur(imgGray, (3, 3), sigmaX=0)
    #     # imgBlur = cv2.medianBlur(imgGray, 3)
    #     ret, binary = cv2.threshold(imgGray, 245, 255, cv2.THRESH_BINARY)
    #
    #     binary = cv2.bitwise_not(binary)
    #     # mor = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((50, 50), np.uint8))
    #
    #     # mor = cv2.erode(binary, kernel=np.ones((3, 3), np.uint8), iterations=7)
    #     # mor = cv2.dilate(mor, kernel=np.ones((3, 3), np.uint8), iterations=7)
    #     # imgCanny = cv2.Canny(binary, 100, 100)
    #     # imgBlank = np.zeros_like(img)
    #
    #     getContours(binary, imgContour)
    #     # getContours(mor, imgContour1)
    #
    #     imgStack = stackImages(0.5, ([imgContour, ],))
    #     # new_filename = os.path.join(root, 'new_' + name)
    #     # cv2.imwrite(new_filename, imgStack)
    #
    #     cv2.imshow('test', imgStack)
    #     cv2.waitKey(0)
    img0 = cv_load_image('http://192.168.7.29:8081/334,0b3b1f438b259f1b')
    orig_img0 = img0.copy()
    a = torch.load('/data4t/data/zhangyufei/data/pk0.pth', map_location='cpu')
    print('torch load: ', a)
    b = a[0].bbox.numpy().tolist()
    print('num of proposal: ', len(b))
    for c in b:
        cv2.rectangle(img0, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 0, 255), 1)

    img2 = cv_load_image('http://192.168.7.13:8081/82,0b3b1fbc39e81393')
    orig_img2 = img2.copy()
    a = torch.load('/data4t/data/zhangyufei/data/pk2.pth', map_location='cpu')
    print('torch load: ', a)
    b = a[0].bbox.numpy().tolist()
    print('num of proposal: ', len(b))
    for c in b:
        cv2.rectangle(img2, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 0, 255), 1)

    img7 = cv_load_image('http://192.168.7.5:8082/195,0b3b1fde37060e59')
    orig_img7 = img7.copy()
    a = torch.load('/data4t/data/zhangyufei/data/pk7.pth', map_location='cpu')
    print('torch load: ', a)
    b = a[0].bbox.numpy().tolist()
    print('num of proposal: ', len(b))
    for c in b:
        cv2.rectangle(img7, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 0, 255), 1)

    img8 = cv_load_image('http://192.168.7.5:8081/102,0b3b1fccf2683854')
    orig_img8 = img8.copy()
    a = torch.load('/data4t/data/zhangyufei/data/pk8.pth', map_location='cpu')
    print('torch load: ', a)
    b = a[0].bbox.numpy().tolist()
    print('num of proposal: ', len(b))
    for c in b:
        cv2.rectangle(img8, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 0, 255), 1)

    img9 = cv_load_image('http://192.168.7.19:8084/284,0b3b1ff6280cf4b5')
    orig_img9 = img9.copy()
    a = torch.load('/data4t/data/zhangyufei/data/pk9.pth', map_location='cpu')
    print('torch load: ', a)
    b = a[0].bbox.numpy().tolist()
    print('num of proposal: ', len(b))
    for c in b:
        cv2.rectangle(img9, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 0, 255), 1)

    imgStack = stackImages(1.0, (
    [orig_img0, img0], [orig_img2, img2], [orig_img7, img7], [orig_img8, img8], [orig_img9, img9],))

    new_filename = os.path.join(root, 'question.jpg')
    print(new_filename)
    cv2.imwrite(new_filename, imgStack)

    cv2.imshow('test', imgStack)
    cv2.waitKey(0)
