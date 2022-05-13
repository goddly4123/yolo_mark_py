import os.path
import copy
from util import *

import cv2
import math
import time
import shutil
import queue
from threading import Thread
import numpy as np

class draw:
    def __init__(self, img_dir):
        self.dir = img_dir
        self.height = None
        self.Default_img = None

        # '/media/nongshim/New_disk_name/Project/best_before/img_num'
        self.max_count = len(os.listdir(img_dir))
        self.TRASH = TRASH()  # 휴지통 이름 선언
        self.dir_list = DEL_DS_Store(img_dir, '_Store')


        self.fontpath = "./icon/NanumGothicLight.ttf"
        self.temp_img = './icon/temp.jpg'
        self.temp_mark_img = './icon/temp_mark.jpg'
        self.num = 0
        self.detection_memory = -100
        self.max_fps = 200

        self.img_memory = None
        self.refresh = False
        self.click_start = False
        self.dont_quit = False
        self.box_line_bold = 1
        self.distance_limit = 0.005
        self.overlap = []
        self.overlap_detected = False

        option = OPTION('read', 1)
        '''설정값 정의'''
        if option['show_grid'] == 'True':  # for mouse position info, show grid
            self.show_grid = True
        else:
            self.show_grid = False

        if option['last_label'] == 'True':  # The last selected value is the default label
            self.last_label = True
        else:
            self.last_label = False

        if option['not_marked'] == 'True':  # If there is no marked data, Don't Save TXT files
            self.not_marked = True
        else:
            self.not_marked = False

        if option['Search_and_jump'] == 'True':  # Searching not marked img and jump
            self.Search_and_jump = True
        else:
            self.Search_and_jump = False

        if option['drag_and_jump'] == 'True':  # Draw One Region and jump next page
            self.drag_and_jump = True
        else:
            self.drag_and_jump = False

        self.show_grid_change = True
        self.last_label_change = True
        self.not_marked_change = True
        self.Search_and_jump_change = True
        self.drag_and_jump_change = True
        self.label_change = True

        self.option = option

        self.auto_save_mode = False  # use with self.Search_and_jump

        self.reinforce_enter = False
        self.slide_change = False
        self.slide_show_again = False
        self.CHOOSE_label_mode = False
        self.label_change_mode = False
        self.one_click = False
        self.small_box_tray = []
        self.small_box_idx = 0
        self.small_box_quit = False
        self.sleep_mode = False
        self.pre_x = 0
        self.pre_y = 0

        self.click = []
        self.colors = [
            (62, 117, 221), (71, 173, 112),
            (230, 127, 197), (250, 183, 64),
            (128, 33, 255), (255, 0, 149),
            (0, 255, 149), (101, 255, 255),
            (93, 28, 73), (216, 138, 255),
            (25, 138, 175), (255, 255, 255),
            (62, 117, 201), (30, 30, 30),
            (71, 173, 112), (191, 127, 197),
            (189, 183, 64), (128, 33, 255),
            (255, 0, 149), (0, 255, 149),
            (101, 255, 255), (93, 28, 73),
            (216, 138, 255), (25, 138, 175),
            (255, 255, 255), (30, 30, 30)
        ]

        self.file_list = ''
        self.property = property_()
        print(self.property)

        self.make_background()

        self.img_x = int(self.width * 0.0153)
        self.img_y = int(self.height * 0.023)
        self.img_w = self.img_x + int(self.width * .54)
        self.img_h = self.img_y + int(self.height * 0.8)
        print('Image position : x-{}. y-{}. w-{}. h-{}'.format(self.img_x, self.img_y, self.img_w, self.img_h))

    def make_background(self):
        self.width, self.height, self.Default_img = get_screen()
        self.load_icon()
        self.Default_img = merge_image(self.Default_img, self.head, int(self.width * 0.66), int(self.height * 0.4))
        self.Default_img = cv2.cvtColor(self.Default_img, cv2.COLOR_RGB2GRAY)
        self.Default_img = cv2.cvtColor(self.Default_img, cv2.COLOR_GRAY2BGR)
        for i in range(15):
            self.Default_img = cv2.GaussianBlur(self.Default_img, (5, 5), 3)

        '''라운드 박스 그리기 모음'''
        self.Default_img = cv2.rectangle(self.Default_img, (int(self.width * 0.01), int(self.height * 0.013)),
                                         (int(self.width * 0.56), int(self.height * 0.831)),
                                         (60, 60, 60), 1)
        self.Default_img = cv2.rectangle(self.Default_img, (int(self.width * 0.01), int(self.height * 0.862)),
                                         (int(self.width * 0.992), int(self.height * 0.978)),
                                         (60, 60, 60), 1)
        self.Default_img = merge_image(self.Default_img, self.bigbox,
                                       int(self.width * 0.583), int(self.height * 0.013))
        self.Default_img = merge_image(self.Default_img, self.mark,
                                       int(self.width * 0.645), int(self.height * 0.702))

        '''top 박스 그리기'''
        self.Default_img = merge_image(self.Default_img, self.topbox,
                                       int(self.width * 0.588), int(self.height * 0.019))
        if self.property['mode'] == 'RCNN':
            RCNN = [(50, 120, 250, 3), 1]
            CNN = [(100, 100, 100, 3), 0]
        else:
            RCNN = [(100, 100, 100, 3), 0]
            CNN = [(50, 200, 100, 3), 1]

        self.Default_img = insert_TEXT(img=self.Default_img, textsize=int(self.width * 0.014),
                                       x=int(self.width * 0.627), y=int(self.height * 0.038),
                                       text='RCNN. Detection Mode',
                                       color=RCNN[0], stroke=RCNN[1])
        self.Default_img = insert_TEXT(img=self.Default_img, textsize=int(self.width * 0.014),
                                       x=int(self.width * 0.795), y=int(self.height * 0.038),
                                       text='CNN. Classification Mode',
                                       color=CNN[0], stroke=CNN[1])
        self.Default_img = self.RCNN(self.Default_img)


    def SLIDE_show2(self, img, back_img, start_point):
        position = self.num / self.max_count
        A = 0.6137
        B = 0.9511
        x = int(self.width * ((B - A) * position + A))
        y = int(self.height * 0.645)

        '''슬라이드바 백그라운드 이미지 덮어쓰기'''
        img = merge_image(img, back_img, start_point[0], start_point[1])

        '''슬라이드 바 만들기'''
        img = merge_image(img, self.slider_bar,
                          int(self.width * 0.5969),
                          int(self.height * 0.6565))

        img = merge_image(img, self.slider_Button, x, y)
        self.button_position = [x / self.width, y / self.height,
                                x / self.width + (0.6115 - 0.6036),
                                y / self.height + (0.6833 - 0.6472)]
        if self.label_change:
            pass

        if 0.600 < self.X < 0.9821 and 0.639 < self.Y < 0.6962:
            temp_num = self.slide_tracking() + 1
            posi_x = int(self.X * self.width - self.slider_Button.shape[1] / 2)
            posi_y = int(self.height * 0.645)

            img = merge_image(img,
                              cv2.subtract(self.slider_Button,
                                           np.full(self.slider_Button.shape, (200, 200, 200, 50),
                                                   dtype=np.uint8)),
                              posi_x, posi_y)

            if temp_num < 0:
                temp_num = 0
            if temp_num > self.max_count - 1:
                temp_num = self.max_count - 1
            img = insert_TEXT(img=img, textsize=int(self.width * 0.009),
                              x=posi_x, y=posi_y + int(self.height * 0.045),
                              text='{} page'.format(temp_num + 1),
                              color=(220, 220, 220, 3), stroke=0)

        return img

    def slide_tracking(self):
        position = self.num / self.max_count
        A = 0.6137
        B = 0.9511
        x = int(self.width * ((B - A) * position + A))
        modify = ((x / self.width + (0.6115 - 0.6036)) - (x / self.width)) / 2

        if self.X > 0.9680:
            self.X = 0.9680
        if self.X < A - 0.008:
            self.X = A - 0.008

        return int((self.max_count * (self.X - modify - A)) / (B - A))

    def load_icon(self):
        self.arrow_down = cv2.imread('./icon/arrow.png', cv2.IMREAD_UNCHANGED)
        self.arrow_down = cv2.resize(self.arrow_down, dsize=(0, 0), fx=.65, fy=.65)

        self.arrow_up = cv2.flip(self.arrow_down, 0)

        self.head = cv2.imread('./icon/head.png', cv2.IMREAD_UNCHANGED)
        self.head = cv2.resize(self.head, dsize=(0, 0), fx=.55, fy=.55)

        self.bigbox = cv2.imread('./icon/bigbox.png', cv2.IMREAD_UNCHANGED)
        self.bigbox = cv2.resize(self.bigbox, (int(self.width * 0.406), int(self.height * 0.8157)))

        self.topbox = cv2.imread('./icon/top_box.png', cv2.IMREAD_UNCHANGED)
        self.topbox = cv2.resize(self.topbox, (int(self.width * 0.397), int(self.height * 0.06)))

        self.mark = cv2.imread('./icon/mark.png', cv2.IMREAD_UNCHANGED)
        self.mark = cv2.resize(self.mark, (int(self.width * 0.27), int(self.height * 0.125)))

        self.left = cv2.imread('./icon/left.png', cv2.IMREAD_UNCHANGED)
        self.left = cv2.resize(self.left, dsize=(0, 0), fx=.9, fy=.9)

        self.right = cv2.imread('./icon/right.png', cv2.IMREAD_UNCHANGED)
        self.right = cv2.resize(self.right, dsize=(0, 0), fx=.9, fy=.9)

        self.slider_bar = cv2.imread('./icon/slide_bar.png', cv2.IMREAD_UNCHANGED)
        self.slider_bar = cv2.resize(self.slider_bar, (int(self.width * 0.38), int(self.height * 0.013)))

        self.slider_Button = cv2.imread('./icon/slider_button.png', cv2.IMREAD_UNCHANGED)

    def simple_RCNN(self, img):
        text_size = .009
        if self.show_grid_change:
            img[int(self.height * 0.4935):int(self.height * 0.5185),
            int(self.width * 0.6589):int(self.width * 0.6859)] = self.sub_img_grid
            if self.show_grid:
                img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                                  x=int(self.width * 0.6618), y=int(self.height * 0.50),
                                  text='ON',
                                  color=(50, 120, 250, 3), stroke=1)
            else:
                img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                                  x=int(self.width * 0.6618), y=int(self.height * 0.50),
                                  text='OFF',
                                  color=(220, 220, 220, 3), stroke=0)
            self.show_grid_change = False

        if self.not_marked_change:
            img[int(self.height * 0.4481):int(self.height * 0.4843),
            int(self.width * 0.8031):int(self.width * 0.8297)] = self.sub_img_not_marked
            if self.not_marked:
                img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                                  x=int(self.width * 0.8058), y=int(self.height * (0.50 - 0.04 * 1)),
                                  text='ON',
                                  color=(50, 120, 250, 3), stroke=1)
            else:
                img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                                  x=int(self.width * 0.8058), y=int(self.height * (0.50 - 0.04 * 1)),
                                  text='OFF',
                                  color=(220, 220, 220, 3), stroke=0)
            self.not_marked_change = False

        if self.last_label_change:
            img[int(self.height * 0.413):int(self.height * 0.4426),
            int(self.width * 0.7854):int(self.width * 0.812)] = self.sub_img_default_label
            if self.last_label:
                img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                                  x=int(self.width * 0.787), y=int(self.height * (0.50 - 0.04 * 2)),
                                  text='ON',
                                  color=(50, 120, 250, 3), stroke=1)
            else:
                img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                                  x=int(self.width * 0.787), y=int(self.height * (0.50 - 0.04 * 2)),
                                  text='OFF',
                                  color=(220, 220, 220, 3), stroke=0)
            self.last_label_change = False

        if self.Search_and_jump_change:
            img[int(self.height * 0.3741):int(self.height * 0.4074),
            int(self.width * 0.7677):int(self.width * 0.7922)] = self.sub_img_search
            if self.Search_and_jump:
                img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                                  x=int(self.width * 0.769), y=int(self.height * (0.50 - 0.04 * 3)),
                                  text='ON',
                                  color=(50, 120, 250, 3), stroke=1)
            else:
                img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                                  x=int(self.width * 0.769), y=int(self.height * (0.50 - 0.04 * 3)),
                                  text='OFF',
                                  color=(220, 220, 220, 3), stroke=0)
            self.Search_and_jump_change = False

        if self.drag_and_jump_change:
            img[int(self.height * 0.3352):int(self.height * 0.3639),
            int(self.width * 0.7734):int(self.width * 0.7953)] = self.sub_img_jump
            if self.drag_and_jump:
                img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                                  x=int(self.width * 0.776), y=int(self.height * (0.50 - 0.04 * 4)),
                                  text='ON',
                                  color=(50, 120, 250, 3), stroke=1)
            else:
                img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                                  x=int(self.width * 0.776), y=int(self.height * (0.50 - 0.04 * 4)),
                                  text='OFF',
                                  color=(220, 220, 220, 3), stroke=0)
            self.drag_and_jump_change = False

        if self.label_change:
            img[int(self.height * 0.1778):int(self.height * 0.2796),
            int(self.width * 0.5875):int(self.width * 0.9776)] = self.sub_img_label
            img = insert_TEXT(img=img, textsize=int(self.width * text_size * 3.5),
                              x=int(self.width * 0.6018), y=int(self.height * .2),
                              text="Label is [ {} ]".format(self.property['target']),
                              color=(220, 220, 220, 3), stroke=2)
            self.label_change = False
        return img

    def RCNN(self, img):
        text_size = .009
        '''top 박스 그리기'''
        img = cv2.line(img, (int(self.width * 0.5982), int(self.height * 0.54)),
                       (int(self.width * 0.9696), int(self.height * 0.54)),
                       (60, 60, 60), 1)

        img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                          x=int(self.width * 0.6018), y=int(self.height * 0.50),
                          text='5. Show grid : ',
                          color=(220, 220, 220, 3), stroke=0)

        img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                          x=int(self.width * 0.6018), y=int(self.height * (0.50 - 0.04 * 1)),
                          text='4. If there is no marked data, Dont Save TXT files : ',
                          color=(220, 220, 220, 3), stroke=0)

        img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                          x=int(self.width * 0.6018), y=int(self.height * (0.50 - 0.04 * 2)),
                          text='3. The last selected value is the default label : ',
                          color=(220, 220, 220, 3), stroke=0)

        img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                          x=int(self.width * 0.6018), y=int(self.height * (0.50 - 0.04 * 3)),
                          text='2. Searching not marked img and jump : ',
                          color=(220, 220, 220, 3), stroke=0)

        img = insert_TEXT(img=img, textsize=int(self.width * text_size),
                          x=int(self.width * 0.6018), y=int(self.height * (0.50 - 0.04 * 4)),
                          text='1. Draw One Region and jump next page : ',
                          color=(220, 220, 220, 3), stroke=0)

        return img

    def load_img(self, num, y, x):
        self.file_list = []
        for i in os.listdir(self.dir):
            if i[-3:] == 'jpg' and i[0] != '.':
                self.file_list.append(i)
        # self.file_list = os.listdir(os.path.join('./img', self.property['target']))
        self.file_list.sort()
        self.max_count = len(self.file_list)
        _ = DEL_DS_Store(self.dir, 'DS_Store')


        print(len(self.file_list), num)

        file_directory = os.path.join(self.dir, self.file_list[num])
        img = cv2.imread(file_directory, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (x, y))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def navigator(self, num, mg_img):
        # print('NUM : ', num, 'MAX : ', self.max_count)
        last = False
        temp_num = num
        if self.max_count - num < 8:
            num = self.max_count - 8
            last = True

        tray = []
        temp = [-6, -5, -4, -3, -2, -1, 0]
        img_tray = ['None']*14

        for i in range(7):
            if num + temp[i] >= 0:
                tray.append(num + temp[i])

        if not tray:
            tray = range(0, 14, 1)
        else:
            for i in range(len(temp) * 2 - len(tray)):
                tray.append(tray[-1] + 1)

        Q = queue.Queue()
        P = []

        for i in range(len(tray)):  #num, y, x, height, dir, Q
            P.append(Thread(target=load_img_navi, args=(tray[i], int(self.height * .1), int(self.height * .1), self.height, self.dir, Q, i, self.file_list)))
            P[i].start()

        while 'None' in img_tray:
            img_ = Q.get()
            img_tray[img_[0]] = img_[1]

        for i in range(len(tray)):
            P[i].join()

        if last:
            num = temp_num

        resent = tray.index(num)

        for idx, img in enumerate(img_tray):
            x = self.img_x + int(self.width * .0699 * idx)
            y = int(self.height * 0.87)
            text = '{} / {} p'.format(tray[idx] + 1, self.max_count)

            if self.max_count > 1000:
                font_size = .007
            else:
                font_size = .009

            '''페이지 번호 표시하기'''
            mg_img = insert_TEXT(img=mg_img, textsize=int(self.width * font_size),
                                 x=x, y=int(self.height * 0.84),
                                 text=text, color=(150, 150, 150, 3), stroke=0)

            '''메인 이미지와 동일한 썸네일에 표시 강화하기'''
            if idx == resent:
                mg_img = merge_image(mg_img, cv2.subtract(img,
                                                          np.full(img.shape, (20, 10, 0),
                                                                  dtype=np.uint8)), x, y)
                mg_img = merge_image(mg_img, self.arrow_down,
                                     int(int(self.width * 0.037) + int(self.width * .0695 * idx)),
                                     int(y * .99))
                mg_img = merge_image(mg_img, self.arrow_up,
                                     int(int(self.width * 0.037) + int(self.width * .0695 * idx)),
                                     int(self.height * .955))
            else:
                mg_img = merge_image(mg_img, img, x, y)

        '''좌우 화살표 표시'''
        mg_img = merge_image(mg_img, self.left, int(self.width * 0.006), int(self.height * .898))
        mg_img = merge_image(mg_img, self.right, int(self.width * 0.975), int(self.height * .898))

        return mg_img

    def mouse_event(self, event, x, y, flags, param):
        # print('EVENT : {}'.format(event))
        _, _ = flags, param
        self.X = round(x / self.width, 4)
        self.Y = round(y / self.height, 4)

        # self.sleep_mode
        self.sleep_mode = False

        move_x = (self.pre_x - x) * (self.pre_x - x)
        move_y = (self.pre_y - y) * (self.pre_y - y)

        move = math.sqrt(move_x + move_y)



        if self.property['mode'] == 'RCNN':
            #if self.X > 0.5863 or self.Y > 0.8686:
            #    self.one_click = False
            #    self.click = []

            if self.CHOOSE_label_mode:
                if event == cv2.EVENT_LBUTTONDOWN:
                    for idx, coordinate in enumerate(self.P.coordinates):
                        if coordinate[0] <= x <= coordinate[2] and coordinate[1] <= y <= coordinate[3]:
                            self.P.detection[-1:][0][0] = idx
                            self.P.quit = True
                            self.one_click = False
                            self.label_change = True

                            if self.last_label:
                                self.property['target'] = self.property['label'][idx]
                                # print(self.property['target'], '[[[[[[')
                                self.slide_change = True

            elif self.label_change_mode:
                if event == cv2.EVENT_LBUTTONDOWN:
                    for idx, coordinate in enumerate(self.coordinates):
                        if coordinate[0] <= x <= coordinate[2] and coordinate[1] <= y <= coordinate[3]:
                            print(self.P.detection[self.small_box_idx])
                            self.P.detection[self.small_box_idx][0] = idx
                            self.small_box_quit = True
                            self.detection_memory = -100
                            self.label_change = True

                            if self.last_label:
                                self.property['target'] = self.property['label'][idx]
                                self.slide_change = True

            else:
                if self.property['mode'] == 'RCNN':
                    if 0.5804 < self.X < 0.9999 and 0.0162 < self.Y < 0.8505:
                        self.slide_change = True
                        self.slide_show_again = True
                    else:
                        self.slide_change = False

                if event == cv2.EVENT_RBUTTONDOWN:
                    '''클릭된 좌표값 생성 및 YOLO_mark좌표로 변환 및 메모리 저장'''
                    if 0 <= x <= self.width * 0.5828 and 0 <= y <= self.height * 0.8630:
                        if self.img_x >= x:
                            x = 0
                        elif self.img_w <= x:
                            x = self.img_w - self.img_x
                        else:
                            x = x - self.img_x

                        if self.img_y >= y:
                            y = 0
                        elif self.img_h <= y:
                            y = self.img_h - self.img_y
                        else:
                            y = y - self.img_y


                        if len(self.click) == 0:
                            pass

                        else:
                            self.click.append([x, y])
                            self.P.ENCODE(self.click)
                            print(self.click)
                            self.CHOOSE_label_mode = True
                            self.label_change = True

                            if self.drag_and_jump:
                                self.auto_save_mode = True

                if event == cv2.EVENT_LBUTTONDOWN:
                    print('left X : {} px,  Y : {} px'.format(self.X, self.Y))

                    if 0.9703 < self.X < 0.9979 and 0.8917 < self.Y < 0.9481:
                        if not self.Search_and_jump:
                            self.num += 13
                            self.detection_memory = -1000
                            self.refresh = True
                        else:
                            i = self.num + 1
                            name = self.file_list[i][:-4] + ".txt"
                            file_name = os.path.join(self.dir, name)
                            if not os.path.exists(file_name):
                                while True:
                                    i += 1
                                    try:
                                        file_name = os.path.join(self.dir, self.file_list[i][:-4] + ".txt")
                                    except:
                                        self.num = i - 1
                                        self.refresh = True
                                        break
                                    if os.path.exists(file_name):
                                        self.num = i - 1
                                        self.refresh = True
                                        break
                            else:
                                while True:
                                    i += 1
                                    lines = []
                                    try:
                                        file_name = os.path.join(self.dir, self.file_list[i][:-4] + ".txt")
                                        with open(file_name, "r") as f:
                                            lines = f.readlines()
                                        if lines == []:
                                            self.num = i - 1
                                            self.refresh = True
                                            break
                                    except:
                                        self.num = i - 1
                                        self.refresh = True
                                        break

                    if 0.0036 < self.X < 0.0245 and 0.8917 < self.Y < 0.9454:
                        if not self.Search_and_jump:
                            self.num -= 13
                            self.detection_memory = -1000
                            self.refresh = True
                        else:
                            i = self.num - 1
                            name = self.file_list[i][:-4] + ".txt"
                            file_name = os.path.join(self.dir, name)
                            if not os.path.exists(file_name):
                                while True:
                                    i -= 1
                                    try:
                                        file_name = os.path.join(self.dir, self.file_list[i][:-4] + ".txt")
                                    except:
                                        self.num = i - 1
                                        self.refresh = True
                                        break
                                    if os.path.exists(file_name):
                                        self.num = i - 1
                                        self.refresh = True
                                        break
                            else:
                                while True:
                                    i -= 1
                                    try:
                                        file_name = os.path.join(self.dir, self.file_list[i][:-4] + ".txt")
                                    except:
                                        self.num = i - 1
                                        self.refresh = True
                                        break
                                    if not os.path.exists(file_name):
                                        self.num = i - 1
                                        self.refresh = True
                                        break

                    if 0.6016 < self.X < 0.6557 and 0.4991 < self.Y < 0.5241:
                        if self.property['mode'] == 'RCNN':
                            if self.show_grid:
                                self.show_grid = False
                            else:
                                self.show_grid = True
                            self.show_grid_change = True

                    if 0.6016 < self.X < 0.7708 and 0.338 < self.Y < 0.3648:
                        if self.property['mode'] == 'RCNN':
                            if self.drag_and_jump:
                                self.drag_and_jump = False
                            else:
                                self.drag_and_jump = True
                            self.drag_and_jump_change = True

                    if 0.6016 < self.X < 0.762 and 0.3787 < self.Y < 0.4009:
                        if self.property['mode'] == 'RCNN':
                            if self.Search_and_jump:
                                self.Search_and_jump = False
                            else:
                                self.Search_and_jump = True
                            self.Search_and_jump_change = True

                    if 0.6016 < self.X < 0.7807 and 0.4185 < self.Y < 0.4398:
                        if self.property['mode'] == 'RCNN':
                            if self.last_label:
                                self.last_label = False
                            else:
                                self.last_label = True
                            self.last_label_change = True

                    if 0.6016 < self.X < 0.8005 and 0.4583 < self.Y < 0.4815:
                        if self.property['mode'] == 'RCNN':
                            if self.not_marked:
                                self.not_marked = False
                            else:
                                self.not_marked = True
                            self.not_marked_change = True

                    '''클릭된 좌표값 생성 및 YOLO_mark좌표로 변환 및 메모리 저장'''
                    if 0 <= x <= self.width * 0.5828 and 0 <= y <= self.height * 0.8630:
                        #x = x - self.img_x
                        #y = y - self.img_y

                        if self.img_x >= x:
                            x = 0
                        elif self.img_w <= x:
                            x = self.img_w - self.img_x
                        else:
                            x = x - self.img_x

                        if self.img_y >= y:
                            y = 0
                        elif self.img_h <= y:
                            y = self.img_h - self.img_y
                        else:
                            y = y - self.img_y

                        if not self.one_click:
                            self.click.append(self.property['target'])
                            self.click.append([x, y])
                            self.one_click = True

                        else:
                            self.click.append([x, y])
                            self.P.ENCODE(self.click)
                            print(self.click)
                            self.click = []
                            self.one_click = False

                            if self.drag_and_jump:
                                self.auto_save_mode = True

                    '''슬라이드바 위치 결정'''
                    if 0.600 < self.X < 0.9821 and 0.639 < self.Y < 0.6962:
                        if not self.refresh:
                            self.num = self.slide_tracking()
                            if self.num + 1 > self.max_count:
                                self.num = self.max_count - 1
                            print(self.num)
                            self.refresh = True

                    '''RCNN / CNN 결정하기'''
                    if 0.7911 < self.X < 0.9506 and 0.0352 < self.Y < 0.0667:
                        self.property['mode'] = 'CNN'
                        print(self.property['mode'])
                        self.num = -1
                        self.make_background()

                        self.refresh = True

            self.pre_x = x
            self.pre_y = y
        elif self.property['mode'] == 'CNN':
            if 0.6232 < self.X < 0.7732 and 0.0352 < self.Y < 0.0667:
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.property['mode'] = 'RCNN'
                    self.num = -1
                    print(self.property['mode'])
                    self.make_background()
                    self.refresh = True

            '''슬라이드바 위치 결정'''
            if 0.600 < self.X < 0.9821 and 0.639 < self.Y < 0.6962:
                if event == cv2.EVENT_LBUTTONDOWN:
                    if not self.refresh:
                        self.num = self.slide_tracking()
                        if self.num + 1 > self.max_count:
                            self.num = self.max_count - 1
                        print(self.num)
                        self.refresh = True

    def right_key(self):
        if self.reinforce_enter:
            x = int(self.X * self.width)
            y = int(self.Y * self.height)
            self.reinforce_enter = False

            if len(self.small_box_tray) != 0 and not self.one_click:
                QUIT = False
                for idx, coordinate in enumerate(self.small_box_tray):
                    if not QUIT:
                        if coordinate[0] <= x <= coordinate[2] and coordinate[1] <= y <= coordinate[3]:
                            self.small_box_idx = idx
                            self.label_change_mode = True
                            self.label_change = True
                            sub_img = self.mg_img[coordinate[1] + 1:coordinate[3] - 1,
                                      coordinate[0] + 1:coordinate[2] - 1]
                            blur_num = 10
                            for i in range(blur_num):
                                self.mg_img = cv2.GaussianBlur(self.mg_img, (5, 5), 3)
                                self.mg_img[coordinate[1] + 1:coordinate[3] - 1,
                                coordinate[0] + 1:coordinate[2] - 1] = sub_img

                                if i == blur_num - 1:
                                    self.mg_img = cv2.rectangle(self.mg_img, (coordinate[0], coordinate[1]),
                                                                (coordinate[2], coordinate[3]),
                                                                (200, 200, 200), 1)
                                cv2.imshow(self.window_name, self.mg_img)
                                cv2.waitKey(5)

                            self.mg_img, self.coordinates = self.P.WINDOW(self.mg_img, [coordinate[0], coordinate[1]])

                            while True:
                                cv2.imshow(self.window_name, self.mg_img)
                                key = cv2.waitKey(20)
                                if key == 27 or self.small_box_quit:
                                    self.dont_quit = True
                                    QUIT = True
                                    break
                                if key == 114:
                                    print('del...1')
                                    self.dont_quit = True
                                    self.P.detection.pop(idx)
                                    self.detection_memory = -1000
                                    QUIT = True
                                    break
                            self.small_box_quit = False
                            self.label_change_mode = False

    def check_overlap(self):
        self.distance_limit = 0.005
        self.overlap = []
        self.overlap_detected = False
        i = 0
        while len(self.P.detection) != i:
            _, box_x, box_y, _, _ = self.P.detection[i]
            j = i + 1
            h = j
            for _, box_X, box_Y, _, _ in self.P.detection[j:]:
                distance = MEASURE_distance(box_x, box_y, box_X, box_Y)
                if distance < self.distance_limit:
                    print('Overlap problem : {}s label vs {}s label - distance: {}.'.format(i, h, distance))
                    if i not in self.overlap:
                        self.overlap.append(i)
                    if h not in self.overlap:
                        self.overlap.append(h)
                    self.overlap_detected = True
                h += 1
            i += 1

        self.overlap.sort()

    def draw_box(self, img, first_img):
        if self.property['mode'] == 'RCNN' and \
                self.detection_memory != len(self.P.detection):
            self.small_box_tray = []
            """메인 이미지 초기화"""
            img = merge_image(img, first_img, self.img_x, self.img_y)
            for i in range(len(self.P.detection)):
                box_id, box_x, box_y, box_X, box_Y = self.P.DECODE(i)
                if box_x > box_X:
                    temp = box_X
                    box_X = box_x
                    box_x = temp
                if box_y > box_Y:
                    temp = box_Y
                    box_Y = box_y
                    box_y = temp

                '''박스 그리기'''
                try:
                    if i not in self.overlap:
                        sub_img = img[box_y:box_Y, box_x:box_X]
                        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 10
                        res = cv2.addWeighted(sub_img, 0.75, white_rect, 0.8, 1.0)
                        img[box_y:box_Y, box_x:box_X] = res

                        img = cv2.rectangle(img, (box_x, box_y),
                                            (box_X, box_Y), self.colors[int(box_id)], self.box_line_bold)
                    else:
                        img = cv2.rectangle(img, (box_x, box_y), (box_X, box_Y), (0, 0, 0), -1)
                        self.overlap.pop(self.overlap.index(i))
                except:
                    self.P.detection.pop(i)

            for i in range(len(self.P.detection)):
                box_id, box_x, box_y, box_X, box_Y = self.P.DECODE(i)
                if box_x > box_X:
                    temp = box_X
                    box_x = temp
                if box_y > box_Y:
                    temp = box_Y
                    box_y = temp

                if box_x < self.x_:
                    box_x = self.x_
                if box_y < self.y_:
                    box_y = self.y_
                if box_X > self.w_:
                    box_X = self.w_
                if box_Y > self.h_:
                    box_Y = self.h_

                '''박스 내 번호 마킹하기'''
                sub_img = img[box_y + 2:box_y + 30, box_x + 2:box_x + 30]
                white_rect = np.full(sub_img.shape, self.colors[int(box_id)], dtype=np.uint8)
                res = cv2.addWeighted(sub_img, 0.7, white_rect, 0.3, 1.0)
                img[box_y + 2:box_y + 30, box_x + 2:box_x + 30] = res

                self.small_box_tray.append([box_x + 1, box_y + 1, box_x + 30, box_y + 30])

                img = insert_TEXT(img=img, textsize=int(self.width * 0.01),
                                  x=box_x + 10, y=box_y + 4,
                                  text='{}.'.format(self.property['label'][box_id]),
                                  #  '{}.  {}'.format(i, self.property['label'][box_id])
                                  color=(255, 255, 255), stroke=1)

            self.detection_memory = len(self.P.detection)

        return img

    def draw_grid(self, img):
        """마우스가 특정 영역 안에 있으면 죄표 표시 하기"""
        if 0 <= self.X <= 0.5828:
            if 0 <= self.Y <= 0.8630:
                if self.x_ > self.X * self.width:
                    self.X = self.x_ / self.width
                if self.w_ < self.X * self.width:
                    self.X = self.w_ / self.width

                if self.y_ > self.Y * self.height:
                    self.Y = self.y_ / self.height
                if self.h_ < self.Y * self.height:
                    self.Y = self.h_ / self.height

                if self.one_click:
                    color = (30, 100, 200)
                else:
                    color = (30, 30, 30)
                img = cv2.rectangle(img, (int(self.width * self.X - 5), int(self.height * self.Y - 5)),
                                    (int(self.width * self.X + 5), int(self.height * self.Y + 5)),
                                    color, -1)

                img = cv2.line(img, (int(self.width * self.X), self.y_),
                               (int(self.width * self.X), self.h_),
                               color, 1)
                img = cv2.line(img, (self.x_, int(self.height * self.Y)),
                               (self.w_, int(self.height * self.Y)),
                               color, 1)
        return img

    def main(self):
        mouse_fps = []
        self.overlap_detected = False
        self.window_name = 'Classifier ( ESC : quit )'
        self.detection_memory = -1000
        loop_count = 0
        Quit = False
        first_scene = True
        self.X = 0
        self.Y = 0

        self.main_img_height = int(self.height * 0.8)
        self.main_img_widht = int(self.width * .54)
        self.x_ = int(self.width * 0.0153)
        self.y_ = int(self.height * 0.023)
        self.w_ = self.x_ + self.main_img_widht
        self.h_ = self.y_ + self.main_img_height


        while True:

            self.refresh = False

            '''메인 이미지를 불러오기'''
            #try:
            first_img = self.load_img(self.num, self.main_img_height, self.main_img_widht)
            #except:
            #    print('There is no image.')
            #    break

            try:
                if self.property['mode'] == 'RCNN':
                    self.P = pic(self.x_, self.y_, self.w_, self.h_, self.property)
                    self.P.READ(os.path.join(self.dir, self.file_list[self.num]))
                    self.check_overlap()
            except:
                pass

            mg_img = merge_image(self.Default_img,
                                 first_img,
                                 self.img_x,
                                 self.img_y)

            '''네비게이션 이미지 불러오기'''
            mg_img = self.navigator(self.num, mg_img)

            cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            #cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(self.window_name, self.mouse_event)

            start_point = [int(0.5875 * self.width), int(0.6361 * self.height)]
            bar_crop = mg_img[int(0.6361 * self.height):int(0.7119 * self.height), start_point[0]:int(1 * self.width)]

            mg_img = insert_TEXT(img=mg_img, textsize=int(self.width * 0.009),
                              x=int(self.width * 0.6018), y=int(self.height * 0.6022),
                              text='7. Image Sequence : {} / {} pages'.format(self.num + 1, self.max_count),
                              color=(220, 220, 220, 3), stroke=0)

            file = os.path.join('img/' + self.file_list[self.num])
            mg_img = insert_TEXT(img=mg_img, textsize=int(self.width * 0.009),
                              x=int(self.width * 0.6018), y=int(self.height * 0.5622),
                              text='6. Image name : {}'.format(file),
                              color=(220, 220, 220, 3), stroke=0)

            '''슬라이드 바 그리기'''
            mg_img = self.SLIDE_show2(mg_img, bar_crop, start_point)
            if self.property['mode'] == 'RCNN':
                #mg_img = self.RCNN(mg_img)
                if first_scene:
                    self.sub_img_grid = mg_img[int(self.height * 0.4935):int(self.height * 0.5185),
                                        int(self.width * 0.6589):int(self.width * 0.6859)]
                    self.sub_img_not_marked = mg_img[int(self.height * 0.4481):int(self.height * 0.4843),
                                              int(self.width * 0.8031):int(self.width * 0.8297)]
                    self.sub_img_default_label = mg_img[int(self.height * 0.413):int(self.height * 0.4426),
                                                 int(self.width * 0.7854):int(self.width * 0.812)]
                    self.sub_img_search = mg_img[int(self.height * 0.3741):int(self.height * 0.4074),
                                          int(self.width * 0.7677):int(self.width * 0.7922)]
                    self.sub_img_jump = mg_img[int(self.height * 0.3352):int(self.height * 0.3639),
                                        int(self.width * 0.7734):int(self.width * 0.7953)]
                    self.sub_img_label = mg_img[int(self.height * 0.1778):int(self.height * 0.2796),
                                         int(self.width * 0.5875):int(self.width * 0.9776)]

                mg_img = self.simple_RCNN(mg_img)

            '''농심마크 이미지 백업'''
            sub_img = mg_img[int(self.height * 0.7139):int(self.height * 0.812),
                      int(self.width * 0.651):int(self.width * 0.949)]
            '''맨 상위 RCNN Detection 이미지 백업'''
            sub_img2 = mg_img[int(self.height * 0.0333):int(self.height * 0.0713),
                       int(self.width * 0.6182):int(self.width * 0.7859)]

            temp_img = copy.copy(mg_img)
            first_scene = False

            while True:
                start_time = time.time()
                if not self.sleep_mode:
                    if self.property['mode'] == 'RCNN':

                        mg_img = temp_img  # cv2.imread(self.temp_img)
                        if self.slide_change:
                            '''슬라이드 바 그리기'''
                            mg_img = self.SLIDE_show2(mg_img, bar_crop, start_point)
                            self.slide_change = False

                            # 백업 이미지 덮어씌우기
                            mg_img[int(self.height * 0.7139):int(self.height * 0.812),
                            int(self.width * 0.651):int(self.width * 0.949)] = sub_img
                            mg_img[int(self.height * 0.0333):int(self.height * 0.0713),
                            int(self.width * 0.6182):int(self.width * 0.7859)] = sub_img2
                        else:
                            if self.slide_show_again:
                                '''슬라이드 바 그리기'''
                                self.slide_change = False
                                mg_img[int(self.height * 0.7139):int(self.height * 0.812),
                                int(self.width * 0.651):int(self.width * 0.949)] = sub_img
                                self.slide_show_again = False

                        mg_img = self.simple_RCNN(mg_img)
                        mg_img = self.draw_box(mg_img, first_img)
                        temp_img = copy.copy(mg_img)
                        if self.show_grid:
                            mg_img = self.draw_grid(mg_img)
                    if self.property['mode'] == 'CNN':
                        mg_img = self.SLIDE_show2(mg_img, bar_crop, start_point)

                '''화면 출력 및 키보드 이벤트 처리'''
                if self.property['mode'] == 'RCNN':
                    if not self.sleep_mode:
                        fps = str(round(1 / (time.time() - start_time), 2))
                    else:
                        fps = str(round(loop_count / (time.time() - start_time), 2))
                    mg_img = cv2.rectangle(mg_img, (int(self.width * 0.9214), int(self.height * 0.0944)),
                                           (int(self.width * 1), int(self.height * 0.1352)), (30, 150, 100), -1)
                    mg_img = insert_TEXT(img=mg_img, textsize=int(self.width * 0.01),
                                         x=int(self.width * 0.93), y=int(self.height * 0.1),
                                         text='FPS : {}'.format(fps),
                                         color=(220, 220, 220, 3), stroke=1)



                self.mg_img = mg_img

                if self.property['mode'] == 'RCNN' and self.CHOOSE_label_mode:
                    self.P.CHOOSE_label(self.mg_img, self.click, self.window_name)
                    self.detection_memory -= 10
                    self.CHOOSE_label_mode = False
                    self.one_click = False
                    self.click = []

                if self.auto_save_mode:
                    if self.property['mode'] == 'RCNN':
                        self.P.SAVE_txt(os.path.join(self.dir, self.file_list[self.num]))
                    Quit = False
                    self.auto_save_mode = False
                    break

                self.right_key()
                self.slide_change = True
                self.sleep_mode = False

                cv2.imshow(self.window_name, mg_img)
                key = cv2.waitKey(1)

                if key != -1:
                    print('KEY : {}'.format(key))

                if key == 3 or key == 32 or key == 83:
                    if self.property['mode'] == 'RCNN':
                        if self.not_marked:
                            if len(self.P.detection) != 0:
                                # 2021-08-27-00-42-55_5error.jpg
                                self.P.SAVE_txt(os.path.join(self.dir, self.file_list[self.num]))
                            else:
                                if os.path.exists(os.path.join(self.dir, self.file_list[self.num][:-3] + 'txt')):
                                    os.remove(os.path.join(self.dir, self.file_list[self.num][:-3] + 'txt'))
                        else:
                            self.P.SAVE_txt(os.path.join(self.dir, self.file_list[self.num]))
                    Quit = False
                    break

                elif key == 82:
                    self.box_line_bold += 1
                    self.refresh = True
                    self.num -= 1

                elif key == 84:
                    if self.box_line_bold != 1:
                        self.box_line_bold -= 1
                        self.refresh = True
                        self.num -= 1

                elif key == 2 or key == 81:
                    if self.property['mode'] == 'RCNN':
                        if self.not_marked:
                            if len(self.P.detection) != 0:
                                self.P.SAVE_txt(os.path.join(self.dir, self.file_list[self.num]))
                        else:
                            self.P.SAVE_txt(os.path.join(self.dir, self.file_list[self.num]))
                    if self.num != 0:
                        self.num -= 2
                        Quit = False
                        break

                elif key == 113:
                    Quit = True
                    break

                elif key == 27:
                    if self.one_click:
                        self.one_click = False
                        self.click = []

                elif key == 40 or key == 255 or key == 127:
                    '''현재 페이지 이미지 파일 삭제'''
                    shutil.move(os.path.join(self.dir, self.file_list[self.num]),
                                os.path.join(self.TRASH, self.file_list[self.num]))
                    DEL_IMG(os.path.join(self.dir, self.file_list[self.num]))
                    self.num -= 2
                    Quit = False
                    break

                elif key == 74 or key == 99 or key == 8:
                    print('Back_space key : 마킹 데이터 전체삭제')
                    self.P.DEL_all()

                elif key == 48:
                    try:
                        self.property['target'] = self.property['label'][0]
                        self.label_change = True
                    except:
                        pass

                elif key == 49:
                    try:
                        self.property['target'] = self.property['label'][1]
                        self.label_change = True
                    except:
                        pass

                elif key == 50:
                    try:
                        self.property['target'] = self.property['label'][2]
                        self.label_change = True
                    except:
                        pass

                elif key == 51:
                    try:
                        self.property['target'] = self.property['label'][3]
                        self.label_change = True
                    except:
                        pass

                elif key == 52:
                    try:
                        self.property['target'] = self.property['label'][4]
                        self.label_change = True
                    except:
                        pass

                elif key == 53:
                    try:
                        self.property['target'] = self.property['label'][5]
                        self.label_change = True
                    except:
                        pass

                elif key == 54:
                    try:
                        self.property['target'] = self.property['label'][6]
                        self.label_change = True
                    except:
                        pass

                elif key == 55:
                    self.property['target'] = self.property['label'][7]
                    self.label_change = True

                elif key == 56:
                    try:
                        self.property['target'] = self.property['label'][8]
                        self.label_change = True
                    except:
                        pass

                elif key == 57:
                    try:
                        self.property['target'] = self.property['label'][9]
                        self.label_change = True
                    except:
                        pass

                elif key == 114:
                    self.reinforce_enter = True

                if self.refresh:
                    self.detection_memory = -1
                    break

                loop_count += 1
                if loop_count > 50:
                    loop_count = 0

            self.show_grid_change = True
            self.last_label_change = True
            self.not_marked_change = True
            self.Search_and_jump_change = True
            self.drag_and_jump_change = True
            self.label_change = True
            self.detection_memory = -1

            if self.max_count - 1 > self.num:
                self.num += 1
            if self.num < 0:
                self.num = 0

            if Quit or os.listdir(self.dir) == []:
                break

        self.option['show_grid'] = self.show_grid
        self.option['last_label'] = self.last_label
        self.option['not_marked'] = self.not_marked
        self.option['Search_and_jump'] = self.Search_and_jump
        self.option['drag_and_jump'] = self.drag_and_jump

        OPTION('write', self.option)


if __name__ == '__main__':
    img_dir = 'img'
    D = draw(img_dir)
    D.main()
