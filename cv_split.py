import cv2
import numpy as np
import time


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


# def getContours(img):
#     # RETR_EXTERNAL:检索极端外部轮廓
#     contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 500:
#             # 画出轮廓
#             # (载体,轮廓,-1,颜色,厚度)
#             cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
#             # 计算轮廓的长度
#             peri = cv2.arcLength(cnt, True)
#             # 计算近似拐角点
#             approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
#             # objCor = len(approx)
#             # print(objCor)
#             # 创建边界框
#             x, y, w, h = cv2.boundingRect(approx)
#             cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def foreground(img, bg_pix=230, ignore_pix=20, idx_off=0, pix_off=0, valid_pix=50):
    """
    img, black supposed to be background.
    bg_pix, pixel color below considered as black
    ignore_pix, pixels wider than <ignore_pix> considered as valid.
    valid_pix, blocks wider than <valid_pix>
    """
    st = time.time()

    h, w = img.shape
    tmp = img.copy()
    # print("find foreground", time.time() - st)
    tmp[tmp < bg_pix] = 0
    # cv2.imshow("tmp", tmp)
    black_cnt = h - np.count_nonzero(tmp, axis=0)  # 每一列有多少0值[0,0,...,0,99,99,99,0,...,0]
    vertical = np.where(black_cnt > ignore_pix)[0]
    if len(vertical) == 0:
        print("count_time_foreground_0", tim e.time() - st)
        return []
    vertical_group = [(x[0], x[-1]) for x in consecutive(vertical) if (len(x) > valid_pix)]
    fore_blocks = []

    for k, vg in enumerate(vertical_group):
        v_tmp = tmp[:, vg[0]: vg[1] + 1]
        v_w = v_tmp.shape[1]
        h_black_cnt = v_w - np.count_nonzero(v_tmp, axis=1)

        horizontal = np.where(h_black_cnt > ignore_pix)[0]
        if len(horizontal) == 0:
            continue
        h_segs = [(x[0], x[-1]) for x in consecutive(horizontal) if (len(x) > 20)]
        if len(h_segs) == 0:
            continue
        h_segs = (h_segs[0][0], h_segs[-1][-1])  # 纵轴有多个图像，显示起始和末尾的y轴坐标

        fore_blocks.append((vg[0] + pix_off, h_segs[0], vg[1] + pix_off, h_segs[1], idx_off + k))
    # if len(fore_blocks) <= 1:
    #     return fore_blocks
    #
    # results = [fore_blocks[0]]
    # for k in range(len(fore_blocks) - 1):
    #     b0 = results[-1]
    #     b1 = fore_blocks[k + 1]
    #     if b1[0] - b0[2] < 2:
    #         results[-1] = (b0[0], min(b0[1], b1[1]), b1[2], max(b0[3], b1[3]), b0[4])
    #     else:
    #         results.append(b1)
    # print("count_time_foreground", time.time() - st)
    # return results
    return fore_blocks


img = cv2.imread("/home/zhangyufei/project/test1.jpg")


# img = np.zeros((512, 512), np.uint8)  # 画布
# img[:] = 255
# cv2.rectangle(img, (0, 400), (200, 500), 0, cv2.FILLED)
# cv2.rectangle(img, (0, 0), (200, 300), 0, cv2.FILLED)
# cv2.rectangle(img, (300, 0), (500, 300), 0, cv2.FILLED)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img)
# imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
# imgCanny = cv2.Canny(imgBlur, 50, 50)
# imgContour = img.copy()
# getContours(imgCanny)

# imgStack = stackImages(0.3, ([img, imgGray, imgBlur], [imgCanny, imgContour, imgContour]))
a = foreground(imgGray, bg_pix=230)

for n in a:
    cv2.rectangle(img, n[:2], n[2:4], 100, 5)

imgStack = stackImages(0.4, ([img_org], [img]))

cv2.imshow("Stack", imgStack)

cv2.waitKey(0)
