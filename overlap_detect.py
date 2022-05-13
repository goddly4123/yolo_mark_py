import os.path
from util import *
import cv2
import math
import shutil


class draw:
    def __init__(self):
        img_dir = '/media/nongshim/New_disk_name/Project/best_before/marked_img'   #  '/media/nongshim/New_disk_name/Project/best_before/img_num'
        self.max_count = len(os.listdir(img_dir))
        self.TRASH = TRASH()  # 휴지통 이름 선언
        self.dir_list = DEL_DS_Store(img_dir, '_Store')
        self.dir = img_dir
        self.fontpath = "NanumGothicLight.ttf"
        self.temp_img = './icon/temp.jpg'
        self.temp_mark_img = './icon/temp_mark.jpg'
        self.num = 0
        self.detection_memory = -100
        self.max_fps = 200
        self.key_fps = 1
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
        self.key = -1
        self.click = []
        self.colors = [
            (62, 117, 221), (71, 173, 112),
            (230, 127, 197), (250, 183, 64),
            (128, 33, 255), (255, 0, 149),
            (0, 255, 149), (101, 255, 255),
            (93, 28, 73), (216, 138, 255),
            (25, 138, 175), (255, 255, 255), (30, 30, 30),
            (62, 117, 201),
            (71, 173, 112), (191, 127, 197),
            (189, 183, 64), (128, 33, 255),
            (255, 0, 149), (0, 255, 149),
            (101, 255, 255), (93, 28, 73),
            (216, 138, 255), (25, 138, 175),
            (255, 255, 255), (30, 30, 30)
        ]

        self.file_list = ''
        self.property = property_()



    def load_img(self, num):
        self.file_list = []
        for i in os.listdir(self.dir):
            if i[-3:] == 'jpg' and i[0] != '.':
                self.file_list.append(i)
        # self.file_list = os.listdir(os.path.join('./img', self.property['target']))
        self.file_list.sort()
        self.max_count = len(self.file_list)
        _ = DEL_DS_Store(self.dir, 'DS_Store')
        
        file_directory = os.path.join(self.dir, self.file_list[num])
        print(file_directory)

    def check_overlap(self):
        self.distance_limit = 0.005
        self.overlap = []
        self.overlap_detected = False
        i = 0
        while len(self.P.detection) != i:
            _, box_x, box_y, _, _ = self.P.detection[i]
            j = i+1
            h = j
            for _, box_X, box_Y, _, _  in self.P.detection[j:]:
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

    def main(self):
        while True:
            self.refresh = False
            start_time = None
            '''메인 이미지를 불러오기'''
            self.load_img(self.num)

            try:
                if self.property['mode'] == 'RCNN':
                    self.P = pic(self.x_, self.y_, self.w_, self.h_, self.property)
                    self.P.READ(os.path.join(self.dir, self.file_list[self.num]))
                    # print(np.array(self.P.detection))
                    self.check_overlap()
            except:
                pass

            if self.overlap_detected :
                shutil.move(os.path.join(self.dir, self.file_list[self.num]),
                            os.path.join(self.TRASH, self.file_list[self.num]))
                shutil.move(os.path.join(self.dir, self.file_list[self.num][:-3]+'txt'),
                            os.path.join(self.TRASH, self.file_list[self.num][:-3]+'txt'))
                #DEL_IMG(os.path.join(self.dir, self.file_list[self.num]))

            self.num += 1


if __name__ == '__main__':
    D = draw()
    D.main()
