import sys
sys.path.append("../")
import cv2
import os
from utils_luo.tools import file_tools

def save_img():
    video_path = '/data/BDD100k/bdd100k/videos/train/'
    img_path = '/data/BDD100k/bdd100k/imgs/train/'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    videos = os.listdir(video_path)
    for video_name in videos:
        file_name = video_name.split('.')[0]
        folder_name = img_path + file_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        vc = cv2.VideoCapture(video_path+video_name) #读入视频文件
        c = 0
        i = 0
        rval=vc.isOpened()

        while rval:   #循环读取视频帧
            c = c + 1
            rval, frame = vc.read()
            pic_path = folder_name+'/'
            if rval and c % 5 == 0:
                i = i+1
                cv2.imwrite(pic_path + file_name + '_' + str(i) + '.jpg', frame[:512, :960, :])
                cv2.waitKey(1)
            elif not rval:
                break
        vc.release()
        print('save_success')
        print(folder_name)

save_img()