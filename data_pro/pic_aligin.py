import cv2 as cv
import numpy as np
import time
from skimage import transform as trans
import math

def card_trans_90(img, name):
    start = time.time()
    h, w = img.shape[0], img.shape[1]
    # img = cv.resize(img, (int(w*0.5), int(h*0.5)))
    # h, w = img.shape[0], img.shape[1]

    # 边缘图像生成
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 300, 300)
    cv.imwrite(name + '_canny.jpg', edges)

    # 检测直线
    minLineLength = int(w * 0.3)
    maxLineGap = 10
    lines = cv.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=80, lines=np.array([]),
                           minLineLength=minLineLength, maxLineGap=maxLineGap)
    a, b, c = lines.shape
    lines = np.reshape(lines, (a, 2, int(c / 2)))
    print(lines)
    for i in range(a):
        print(lines)
        print(lines[i])
        dx = abs(lines[i][1][0] - lines[i][0][0])
        dy = abs(lines[i][1][1] - lines[i][0][1])
        if dy > dx:
            cv.line(img, (lines[i][0][0], lines[i][0][1]),
                    (lines[i][1][0], lines[i][1][1]), (0, 0, 255), 3, cv.LINE_AA)
        else:
            lines = np.delete(lines, [i])
    # 变换直线求相似变换矩阵
    fromeline = lines[0].copy()
    toline = lines[0].copy()
    mm = min(toline[0, 0], toline[1, 0])
    toline[0, 0], toline[1, 0] = mm, mm
    tform = trans.SimilarityTransform()  # 引用 class SimilarityTransform()
    cv.line(img, (toline[0][0], toline[0][1]),
            (toline[1][0], toline[1][1]), (0, 255, 0), 3, cv.LINE_AA)
    cv.imwrite(name + '_lines.jpg', img)
    print(fromeline)
    print(toline)
    tform.estimate(fromeline, toline)  # 从一组对应的点估计转换,随便两个点就可以，3个点5个点都可以
    M = tform.params[0:2, :]  # 得到(3, 3) 的齐次变换矩阵

    # 用相似变换矩阵进行整图的仿射变换
    warped = cv.warpAffine(img, M, (w, h), borderValue=0.0)
    cv.imwrite(name + '_trans.jpg', warped)
    print(time.time() - start)


def rotated_img_with_radiation(img, name, is_show=False):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    if is_show:
        cv.imshow('thresh', thresh)
    # 计算包含了旋转文本的最小边框
    coords = np.column_stack(np.where(thresh > 0))

    # 该函数给出包含着整个文字区域矩形边框，这个边框的旋转角度和图中文本的旋转角度一致
    angle = cv.minAreaRect(coords)[-1]
    print(angle)
    # 调整角度
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # 仿射变换
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    print(angle)
    M = cv.getRotationMatrix2D(center, 20, 1.0)
    rotated = cv.warpAffine(gray, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    cv.imwrite(name + '_trans.jpg', rotated)
    if is_show:
        cv.putText(rotated, 'Angle: {:.2f} degrees'.format(angle), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        print('[INFO] angel :{:.3f}'.format(angle))
        cv.imshow('Rotated', rotated)
        cv.waitKey()
    return rotated


if __name__ == '__main__':
    # pname = 'sample'
    # pname = 'WechatIMG75'
    pname = 'WechatIMG76'
    image = cv.imread('/Users/finup/Desktop/rg/face_rg_server_new/' + pname + '.jpg', cv.IMREAD_COLOR)
    # card_trans_90(image, pname)

    rotated_img_with_radiation(image, pname)
