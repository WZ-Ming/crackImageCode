from builtins import dict
import cv2
import numpy as np
from PIL import Image  # , ImageDraw, ImageFont
import math
import os
from typing import Tuple


class CrackExchangeImage(object):

    def __init__(self, row, col, image: np.ndarray) -> None:
        self.col = col
        self.row = row
        self.originImage = image.copy()
        self.greyImage = cv2.cvtColor(self.originImage.copy(), cv2.COLOR_RGB2GRAY)
        self.edgeLength = int(self.greyImage.shape[1] / self.col)

    def _getSortEdgeDict(self, possibleImage):
        rowStandardMinus = []
        colStandardMinus = []
        edgeDict = dict()

        for i in range(self.col):
            minusValueSum = np.uint32(0)  # 初始化为 uint32 类型
            minusValueList = []
            for j in range(self.edgeLength):
                if (possibleImage[self.edgeLength - 1, i * self.edgeLength + j] >=
                        possibleImage[self.edgeLength, i * self.edgeLength + j]):
                    minusValueList.append(possibleImage[self.edgeLength - 1, i * self.edgeLength + j] -
                                          possibleImage[self.edgeLength, i * self.edgeLength + j])
                else:
                    minusValueList.append(possibleImage[self.edgeLength, i * self.edgeLength + j] -
                                          possibleImage[self.edgeLength - 1, i * self.edgeLength + j])
                minusValueSum += minusValueList[-1]
            averageValue = minusValueSum / self.edgeLength
            powValue = 0
            for value in minusValueList:
                powValue += pow(value - averageValue, 2)
            rowStandardMinus.append(math.sqrt(powValue / self.edgeLength))

        for j in range(self.row):
            for i in range(self.col - 1):
                minusValueSum = np.uint32(0)
                minusValueList = []
                for k in range(self.edgeLength):
                    if (possibleImage[j * self.edgeLength + k, self.edgeLength *
                                      (i + 1) - 1] >= possibleImage[j * self.edgeLength + k, self.edgeLength *
                                                                    (i + 1)]):
                        minusValueList.append(possibleImage[j * self.edgeLength + k, self.edgeLength * (i + 1) - 1] -
                                              possibleImage[j * self.edgeLength + k, self.edgeLength * (i + 1)])
                    else:
                        minusValueList.append(possibleImage[j * self.edgeLength + k, self.edgeLength * (i + 1)] -
                                              possibleImage[j * self.edgeLength + k, self.edgeLength * (i + 1) - 1])
                    minusValueSum += minusValueList[-1]
                averageValue = minusValueSum / self.edgeLength
                powValue = 0
                for value in minusValueList:
                    powValue += pow(value - averageValue, 2)
                colStandardMinus.append(math.sqrt(powValue / self.edgeLength))

        edgeDict[0] = (rowStandardMinus[0] + colStandardMinus[0]) / 2
        edgeDict[1] = (rowStandardMinus[1] + colStandardMinus[0] + colStandardMinus[1]) / 3  # rowStandardMinus[1]+
        edgeDict[2] = (rowStandardMinus[2] + colStandardMinus[1] + colStandardMinus[2]) / 3  # rowStandardMinus[2]+
        edgeDict[3] = (rowStandardMinus[3] + colStandardMinus[2]) / 2
        edgeDict[4] = (rowStandardMinus[0] + colStandardMinus[3]) / 2
        edgeDict[5] = (rowStandardMinus[1] + colStandardMinus[3] + colStandardMinus[4]) / 3  # rowStandardMinus[1]+
        edgeDict[6] = (rowStandardMinus[2] + colStandardMinus[4] + colStandardMinus[5]) / 3  # rowStandardMinus[2]+
        edgeDict[7] = (rowStandardMinus[3] + colStandardMinus[5]) / 2

        edgeDict = dict(sorted(edgeDict.items(), key=lambda item: item[1], reverse=True))

        edgeDict0 = edgeDict.copy()
        first_key, first_value = next(iter(edgeDict0.items()))
        if first_key == 0:
            edgeDict0[1] = (edgeDict0[1] * 3 - colStandardMinus[0]) / 2
            edgeDict0[4] = edgeDict0[4] * 2 - rowStandardMinus[0]
        elif first_key == 1:
            edgeDict0[0] = edgeDict0[0] * 2 - colStandardMinus[0]
            edgeDict0[2] = (edgeDict0[2] * 3 - colStandardMinus[1]) / 2
            edgeDict0[5] = (edgeDict0[5] * 3 - rowStandardMinus[1]) / 2
        elif first_key == 2:
            edgeDict0[1] = (edgeDict0[1] * 3 - colStandardMinus[1]) / 2
            edgeDict0[3] = edgeDict0[3] * 2 - colStandardMinus[2]
            edgeDict0[6] = (edgeDict0[6] * 3 - rowStandardMinus[2]) / 2
        elif first_key == 3:
            edgeDict0[2] = (edgeDict0[2] * 3 - colStandardMinus[2]) / 2
            edgeDict0[7] = edgeDict0[7] * 2 - rowStandardMinus[3]
        elif first_key == 4:
            edgeDict0[0] = edgeDict0[0] * 2 - rowStandardMinus[0]
            edgeDict0[5] = (edgeDict0[5] * 3 - colStandardMinus[3]) / 2
        elif first_key == 5:
            edgeDict0[1] = (edgeDict0[1] * 3 - rowStandardMinus[1]) / 2
            edgeDict0[4] = edgeDict0[4] * 2 - colStandardMinus[3]
            edgeDict0[6] = (edgeDict0[6] * 3 - colStandardMinus[4]) / 2
        elif first_key == 6:
            edgeDict0[2] = (edgeDict0[2] * 3 - rowStandardMinus[2]) / 2
            edgeDict0[5] = (edgeDict0[5] * 3 - colStandardMinus[4]) / 2
            edgeDict0[7] = edgeDict0[7] * 2 - colStandardMinus[5]
        elif first_key == 7:
            edgeDict0[3] = edgeDict0[3] * 2 - rowStandardMinus[3]
            edgeDict0[6] = (edgeDict0[6] * 3 - colStandardMinus[5]) / 2
        edgeDict0 = dict(sorted(edgeDict0.items(), key=lambda item: item[1], reverse=True))

        for key in edgeDict.keys():
            edgeDict[key] = edgeDict[key] + edgeDict0[key]
        edgeDict = dict(sorted(edgeDict.items(), key=lambda item: item[1], reverse=True))
        return edgeDict

    def _exchangeImage(self, indexs, isGrey=True):
        pointList = [
            int(indexs[0] % self.col),
            int(indexs[0] / self.col),
            int(indexs[1] % self.col),
            int(indexs[1] / self.col),
        ]
        beginPos = [pointList[0] * self.edgeLength, pointList[1] * self.edgeLength]
        endPos = [pointList[2] * self.edgeLength, pointList[3] * self.edgeLength]

        if isGrey:
            image = self.greyImage
        else:
            image = self.originImage

        image0 = image[beginPos[1]:beginPos[1] + self.edgeLength, beginPos[0]:beginPos[0] + self.edgeLength]
        image1 = image[endPos[1]:endPos[1] + self.edgeLength, endPos[0]:endPos[0] + self.edgeLength]

        newImage = image.copy()
        newImage[beginPos[1]:beginPos[1] + self.edgeLength, beginPos[0]:beginPos[0] + self.edgeLength] = image1
        newImage[endPos[1]:endPos[1] + self.edgeLength, endPos[0]:endPos[0] + self.edgeLength] = image0
        return newImage

    def getExchangePos(self) -> Tuple[int, int]:
        valueDict = dict()
        for i in range(self.col * self.row - 1):
            for j in range(i + 1, self.col * self.row):
                possibleImage = self._exchangeImage((i, j))
                edgeDict = self._getSortEdgeDict(possibleImage)
                valueSum = 0
                for value in edgeDict.values():
                    valueSum += value
                valueDict[valueSum] = (i, j)

        valueDict = dict(sorted(valueDict.items()))
        exchangeIndexs = list(valueDict.values())[0]

        return exchangeIndexs

    def getRestoreImage(self, indexs: Tuple[int, int]) -> np.ndarray:
        return self._exchangeImage(indexs, False)


if __name__ == "__main__":
    image_file = f"{os.path.dirname(__file__)}/exchange.png"
    image = cv2.imread(image_file, cv2.COLOR_RGB2BGR)

    _crack = CrackExchangeImage(2, 4, image)
    i, j = _crack.getExchangePos()
    resoreImage = _crack.getRestoreImage((i, j))

    cv2.imshow("origin", image)
    cv2.imshow("restoreImage", resoreImage)
    cv2.imwrite(f"{os.path.dirname(__file__)}/result1.png", resoreImage)
    cv2.waitKey()
    cv2.destroyAllWindows()