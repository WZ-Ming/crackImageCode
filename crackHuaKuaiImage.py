import cv2
import numpy as np
import os


class CrackHuaKuaiImage:

    @staticmethod
    def getPos(origin_background_image: np.ndarray, origin_jigsaw_image: np.ndarray):
        back_image = origin_background_image.copy()
        jigsaw_image = origin_jigsaw_image.copy()
        jigsaw_morphoImage = cv2.morphologyEx(jigsaw_image, cv2.MORPH_OPEN, np.ones((3, 3), np.int8))
        back_morphoImage = cv2.morphologyEx(back_image, cv2.MORPH_OPEN, np.ones((3, 3), np.int8))
        jigsaw_canny_image = cv2.Canny(jigsaw_morphoImage, 50, 150)
        back_canny_image = cv2.Canny(back_morphoImage, 50, 150)
        result = cv2.matchTemplate(back_canny_image, jigsaw_canny_image, cv2.TM_CCOEFF_NORMED)

        return cv2.minMaxLoc(result)[3]


if __name__ == "__main__":
    back_picPath = f"{os.path.dirname(__file__)}/background.png"
    jigsaw_picPath = f"{os.path.dirname(__file__)}/slider.png"

    background_image = cv2.imread(back_picPath, cv2.COLOR_RGB2BGR)
    jigsaw_image = cv2.imread(jigsaw_picPath, cv2.COLOR_RGB2BGR)
    pos = CrackHuaKuaiImage.getPos(background_image, jigsaw_image)
    pic = np.hstack([jigsaw_image, background_image])
    cv2.rectangle(
        background_image,
        (pos[0], pos[1]),
        (pos[0] + jigsaw_image.shape[1], pos[1] + jigsaw_image.shape[0]),
        (0, 255, 255),
        2,
    )

    cv2.imshow("pic", pic)
    cv2.imwrite(f"{os.path.dirname(__file__)}/draw_rect.png", background_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
