import os
import numpy as np
import screeninfo
import cv2
from PIL import ImageFont, ImageDraw, Image
import math


def load_img_navi(num, y, x, height, dir, Q, seq, file_list):

    try:
        file_directory = os.path.join(dir, file_list[num])

        img = cv2.imread(file_directory, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (x, y))
        Q.put([seq, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)])
        #print(file_directory)

    except:
        Q.put([seq, np.full((int(height * .1), int(height * .1), 3), 45, np.uint8)])

def get_screen():
    screen = screeninfo.get_monitors()[0]
    print(screen)
    Default_img = np.full((screen.height, screen.width, 3), 30, np.uint8)

    return screen.width, screen.height, Default_img


def DEL_DS_Store(dir_, text):
    """해당 디렉토리 안에서 _Store 문자열을 찾고 """
    str_match = list(filter(lambda x: text in x, os.listdir(dir_)))

    '''삭제하기'''
    for i in str_match:
        os.remove(os.path.join(dir_, i))

    return os.listdir(dir_)


def MEASURE_distance(x, y, X, Y):
    # print(x, y, X, Y)
    move_x = (x - X) * (x - X)
    move_y = (y - Y) * (y - Y)

    move = math.sqrt(move_x + move_y)
    return round(move,4)

def OPTION(activate, option):
    label_list = {}
    if activate == 'read':
        with open('data/option.txt', "r") as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip().split('=')
                label_list[info[0]] = info[1]
        print('Default option : {}'.format(label_list))
        return label_list
    elif activate == 'write':
        with open('data/option.txt', "w") as f:
            for key, value in option.items():
                f.write('{}={}\n'.format(key, value))

def property_():
    """프로그램에 관여하는 속성들의 초기값 선언"""
    '''
       continue_mode : 사전 학습 모델을 활용하여 미리 마킹시키기
       label         : 이미지 폴더내에 현재 보유중인 레이블
       target        : 현재 마킹해야하는 레이블(초기값은 label_list 내 첫번째)
    '''
    label_list = []
    with open('./data/label.txt', "r") as f:
        lines = f.readlines()
        for line in lines:
            info = line.strip().split(' ')
            label_list.append(info[0])

    property__ = {"continue_mode": False,
                  'label': label_list,
                  'target': label_list[0],
                  'mode': 'RCNN'
                  }

    return property__


def merge_image(back, front, x, y) -> object:
    # convert to rgba
    if back.shape[2] == 3:
        back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
    if front.shape[2] == 3:
        front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)

    # crop the overlay from both images
    bh, bw = back.shape[:2]
    fh, fw = front.shape[:2]
    x1, x2 = max(x, 0), min(x + fw, bw)
    y1, y2 = max(y, 0), min(y + fh, bh)
    front_cropped = front[y1 - y:y2 - y, x1 - x:x2 - x]
    back_cropped = back[y1:y2, x1:x2]

    alpha_front = front_cropped[:, :, 3:4] / 255
    alpha_back = back_cropped[:, :, 3:4] / 255

    # replace an area in result with overlay
    result = back.copy()
    result[y1:y2, x1:x2, :3] = alpha_front * front_cropped[:, :, :3] + (1 - alpha_front) * back_cropped[:, :, :3]
    result[y1:y2, x1:x2, 3:4] = (alpha_front + alpha_back) / (1 + alpha_front * alpha_back) * 255

    result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)

    return result


def insert_TEXT(img, textsize, x, y, text, color, stroke):
    font = ImageFont.truetype("./icon/NanumGothicLight.ttf", textsize)
    img_pil: Image = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text,
              font=font, fill=color, stroke_width=stroke)
    return np.array(img_pil)


def TRASH():
    if not os.path.exists('./TRASH'):
        os.mkdir('TRASH')
    return './TRASH'


def DEL_IMG(file):
    try:
        os.remove(file[:-4] + ".txt")
    except:
        pass


class pic:
    def __init__(self, x, y, X, Y, property_1):
        self.file_name = ''
        self.detection = []
        self.x = x
        self.y = y
        self.X = X
        self.Y = Y
        self.x_size = X - x
        self.y_size = Y - y
        self.img = None
        self.property = property_1
        self.coordinates = []
        self.quit = False

    def READ(self, name):
        file_name = name[:-4] + ".txt"
        #print(file_name)
        with open(file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip().split(' ')
                self.detection.append([int(info[0]), float(info[1]), float(info[2]), float(info[3]), float(info[4])])
                # print(info)

    def ENCODE(self, data):
        id_ = data[0]
        id_ = self.property['label'].index(id_)
        start = data[1]
        end = data[2]

        '''start = (x,y), end = (X,Y)'''
        x_center = round(float((end[0] / self.x_size + start[0] / self.x_size) / 2), 4)
        # print('x_center : {}. x : {}, X : {}, size : {}'.format(x_center, start[0], end[0], self.x_size))
        y_center = round(float((end[1] / self.y_size + start[1] / self.y_size) / 2), 4)
        if end[0] - start[0] >= 0:
            x_width = round(float((end[0] - start[0]) / self.x_size), 4)
        else:
            x_width = round(float((start[0] - end[0]) / self.x_size), 4)

        if end[1] - start[1] >= 0:
            y_height = round(float((end[1] - start[1]) / self.y_size), 4)
        else:
            y_height = round(float((start[1] - end[1]) / self.y_size), 4)
        # print('Encording ID : {}, x : {}, y : {}, W : {}, H : {}'.format(id_, x_center, y_center, x_width, y_height))
        self.ADD([id_, x_center, y_center, x_width, y_height])

    def DECODE(self, num):
        id_, x, y, w, h = self.detection[num]
        # print('ID : {}, x : {}, y : {}, W : {}, H : {}'.format(id_, x, y, w, h))
        start_x = int(self.x_size * (x - w / 2))+self.x
        start_y = int(self.y_size * (y - h / 2))+self.y
        end_x = int(self.x_size * (x + w / 2))+self.x
        end_y = int(self.y_size * (y + h / 2))+self.y
        return id_, start_x, start_y, end_x, end_y

    def ADD(self, info):
        """ info = id, x, y, w, h"""
        self.detection.append(info)

    def DEL(self, info):
        idx = self.detection.index(info)
        self.detection.pop(idx)

    def DEL_all(self):
        self.detection = []

    def SAVE_txt(self, name):
        #print(name)
        file_name = name[:-4] + ".txt"
        with open(file_name, "w") as f:
            for label, x, y, w, h in self.detection:
                f.write("{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h))

    def WINDOW(self, img, start_point):
        self.width = img.shape[1]
        self.height = img.shape[0]
        tray = []
        black_box_x = 200
        black_box_y = 20
        x_point = start_point[0] + 5
        y_point = start_point[1] + 5
        margin_height = int(self.height * 0.9)

        # start_point[0], start_point[1] + (idx * (black_box_y + 3))
        count = len(self.property['label'])
        expect_height = y_point + (count * (black_box_y + 5)) + black_box_y
        if margin_height < expect_height:
            y_point = y_point + (margin_height - expect_height)

        for idx, label in enumerate(self.property['label']):
            if idx % 2 == 0:
                black_box_color = 30
            else:
                black_box_color = 50
            input_ = np.full((black_box_y, black_box_x, 3), black_box_color, np.uint8)
            input_ = insert_TEXT(img=input_, textsize=int(self.width * 0.009),
                                 x=3, y=2,
                                 text='{}.{}'.format(idx, label),
                                 color=(220, 220, 220, 3), stroke=0)

            img = merge_image(img, input_, x_point, y_point + (idx * (black_box_y + 3)))
            coordinates = [x_point, y_point + (idx * (black_box_y + 3)),
                           x_point + black_box_x, y_point + (idx * (black_box_y + 3)) + black_box_y]
            tray.append(coordinates)

        return img, tray

    def CHOOSE_label(self, mg_img, data, window_name):
        self.width = mg_img.shape[1]
        self.height = mg_img.shape[0]
        start_point = [data[2][0]+self.x, data[2][1]+self.y]

        x_posi = [data[1][0]+self.x, data[2][0]+self.x]
        y_posi = [data[1][1]+self.y, data[2][1]+self.y]
        x_posi.sort()
        y_posi.sort()
        sub_img = mg_img[y_posi[0] + 1:y_posi[1] - 1, x_posi[0] + 1:x_posi[1] - 1]

        blur_num = 10
        for i in range(blur_num):
            mg_img = cv2.GaussianBlur(mg_img, (5, 5), 3)
            mg_img[y_posi[0] + 1:y_posi[1] - 1, x_posi[0] + 1:x_posi[1] - 1] = sub_img

            if i == blur_num - 1:
                mg_img = cv2.rectangle(mg_img, (x_posi[0], y_posi[0]),
                                       (x_posi[1], y_posi[1]),
                                       (200, 200, 200), 2)
            cv2.imshow(window_name, mg_img)
            cv2.waitKey(1)

        mg_img, self.coordinates = self.WINDOW(mg_img, start_point)

        while True:
            cv2.imshow(window_name, mg_img)
            key = cv2.waitKey(200)
            if self.quit:
                break
            if key == 27:
                if len(self.detection) != 0:
                    self.detection.pop(len(self.detection)-1)
                break

        self.quit = False
