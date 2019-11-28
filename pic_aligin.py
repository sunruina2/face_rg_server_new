import cv2 as cv
import numpy as np
import time
from skimage import transform as trans


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


if __name__ == '__main__':
    # pname = 'sample'
    # pname = 'WechatIMG14076'
    pname = 'WechatIMG75'
    # pname = 'WechatIMG76'
    image = cv.imread('/Users/finup/Desktop/rg/face_rg_server_new/' + pname + '.jpg', cv.IMREAD_COLOR)
    card_trans_90(image, pname)
