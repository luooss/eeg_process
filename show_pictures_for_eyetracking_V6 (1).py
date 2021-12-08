#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from psychopy import visual, core, event, clock, gui, data, parallel
from psychopy.misc import fromFile
import numpy as np
from numpy import *
from PIL import Image
import os
import random
import re
from socket import *
import json
#import pyautogui as pag
import time
import datetime

time_list_s = []
time_list_e = []

port = parallel.ParallelPort(address=0x3EFC)
port.setData(0)


def desample(im):
    w, h = im.size
    w_new = int(w/2)
    h_new = int(h/2)
    out = im.resize((w_new, h_new), Image.ANTIALIAS)
    return out


def trig(num):
    time_trigger_0 = time.time()
    port.setData(num)
    print('setdata time {:.10f}'.format(time.time() - time_trigger_0))
    time_list_e.append(time.time())
    dtimer = core.CountdownTimer(0.02)
    while dtimer.getTime() > 0:
        dddd = 1
    port.setData(0)


def loadData(load_path):
    path_list = []
    # name_list=[]
    for root, parents, name in os.walk(load_path):
        for name_item in name:
            p = root + name_item
            path_list.append(p)
    random.shuffle(path_list)
    return path_list


def task1(save_path, save_name, text, load_path):
    task = 'task1'
    fileName = save_path + save_name
    isExists = os.path.exists(save_path)
    if not isExists:
        os.makedirs(save_path)
    dataFile = open(fileName, 'a')
    # dataFile.write('imageName, left_top_width (pixel), left_top_height (pixel), right_bottom_width (pixel), right_bottom_height (pixel), screen width (pixel),  screen height (pixel)\n')
    dataFile.write(
        'imageName, start_show_time, end_show_time, dlg_time, category, score\n')

    pretext.text = text
    path_list = loadData(load_path)
    pi = 0
    pretext.text = 'BCMI世界名画眼动及脑电采集实验\n请按空格键开始'
    pretext.draw()
    win.flip()

    k_1 = event.waitKeys(keyList=['space', 'escape'])

    time_list_s.append(time.time())

    # print('press', k_1[0])
    # im=Image.open(path_list[pi])
    while pi < len(path_list) and k_1[0] != 'escape':
        num_leave = len(path_list) - pi
        pretext.text = '剩余' + str(num_leave) + '张图片\n按空格键继续'
        pretext.draw()
        win.flip()

        if pi == 0:
            print('load start')
            pic = visual.ImageStim(win, pos=(0, 0))
            #pic.image = path_list[pi]
            try:
                pic.image = Image.open(path_list[pi])
            except:
                print('error!', path_list[pi])
                f = open('error_photo_list.txt', 'a')
                f.write(path_list[pi])
                f.write('\n')
                f.close()
                im = Image.open(path_list[pi])
                im_des = desample(im)
                while 1:
                    try:
                        pic.image = im_des
                    except:
                        im_des = desample(im_des)
                    else:
                        break
            print('load done')
        else:
            pic = pic_tmp

        wi, hi = pic.size
        wp = w*wi/2
        hp = h*hi/2
        # print('图片原始像素大小',wp,hp)
        if (hp/wp) >= (h/w):
            pic.size = (wi/hi*2*rate, 2*rate)
#            print('图片与屏幕比（已乘2）',pic.size)
            h_len_screen = h*rate
            w_len_screen = wp/hp*h*rate

            pix_start = ((w-w_len_screen)/2, (h-h_len_screen)/2)
            pix_end = ((w-w_len_screen)/2+w_len_screen,
                       (h-h_len_screen)/2+h_len_screen)
#            print('开始结束点在屏幕上位置')
        else:
            pic.size = (2*rate, rate*2*hi/wi)
#            print('图片与屏幕比（已乘2）',pic.size)
            h_len_screen = hp/wp*w*rate
            w_len_screen = w*rate
            pix_start = ((w-w_len_screen)/2, (h-h_len_screen)/2)
            pix_end = ((w-w_len_screen)/2+w_len_screen,
                       (h-h_len_screen)/2+h_len_screen)
#            print('开始结束点在屏幕上位置')
        # print(pix_start)
        # print(pix_end)
        shortname = path_list[pi].split('/')[-1]
        # dataFile.write(shortname+', '+str(pix_start[0]) + ', ' + str(pix_start[1]) + ', ' + str(pix_end[0])+', '+str(pix_end[1])+', '+ str(w) + ', '+str(h)+'\n')

        k_1 = event.waitKeys(keyList=['space', 'escape'])  # 开始显示
        print(int(shortname[: shortname.rindex('.')]))

        trig(int(shortname[: shortname.rindex('.')]))

        dataFile.write(shortname+', '+str(datetime.datetime.now().time()))

        time_0 = time.time()
        time_list_s.append(time.time())
        time_1 = time.time()

        time_2 = time.time()
        print('append time', time_1 - time_0)
        print('trigger time', time_2 - time_1)

        #print('press', k_1[0])
        if k_1[0] == 'escape':
            # udpCliSock.sendto('break'.encode("utf-8"), ADDR)
            break
        '''
        socket_dic = {'img_path':path_list[pi], 'left_top_width':str(pix_start[0]), 'left_top_height':str(pix_start[1]), 
        'right_bottom_width':str(pix_end[0]), 'right_bottom_height':str(pix_end[1]), 'screen_w':str(w), 'screen_h':str(h)}
        '''
#        socket_json = json.dumps(socket_dic)
#        udpCliSock.sendto(socket_json.encode('utf-8'), ADDR)

        for ii in range(3):
            pic.draw()
            #print('draw done')
            win.flip()
            #print('flip done')
#        pag.press('s')

        if pi < len(path_list)-1:
            pic_tmp = visual.ImageStim(win, pos=(0, 0))
            try:
                pic_tmp.image = path_list[pi + 1]
            except:
                print('error!', path_list[pi+1])
                f = open('error_photo_list.txt', 'a')
                f.write(path_list[pi+1])
                f.write('\n')
                f.close()
                im = Image.open(path_list[pi+1])
                im_des = desample(im)

                while 1:
                    try:
                        pic_tmp.image = im_des
                    except:
                        im_des = desample(im_des)
                    else:
                        break
        # k_1 = event.waitKeys(keyList = ['space','escape'])#结束显示
        core.wait(20)
        trig(int(shortname[: shortname.rindex('.')]))
        dataFile.write(',' + str(datetime.datetime.now().time()))

        time_list_s.append(time.time())

        win.winHandle.minimize()  # minimise the PsychoPy window
        win.winHandle.set_fullscreen(False)  # disable fullscreen
        win.flip()  # redraw the (minimised) window

        fields = {"情绪类别:": ["正向", "中性", "负向"],
                  "情绪强度:": ["1", "2", "3", "4", "5"]}
        score_dilg = gui.DlgFromDict(fields, title="请您为油画情绪打分", order=["情绪类别:", "情绪强度:"])
        if score_dilg.OK:
            dataFile.write(',' + str(datetime.datetime.now().time()) + ',' + fields["情绪类别:"] + ',' + fields["情绪强度:"]+'\n')
            print(fields)
        else:
            print("User cancelled.")

        win.winHandle.maximize()
        win.winHandle.set_fullscreen(True)
        win.winHandle.activate()
        win.flip()

        # if k_1[0]=='escape':
#            udpCliSock.sendto('break'.encode("utf-8"), ADDR)
        #    break
        pi += 1
    pretext.text = '请按空格键结束实验'
    pretext.draw()
    win.flip()

    k_1 = event.waitKeys(keyList=['space'])
    time_list_s.append(time.time())


if __name__ == '__main__':
    # 被试信息记录窗口

    info = {'pos_save_name': ''}
    infoDlg = gui.DlgFromDict(dictionary=info, title=u'基本信息', order=['pos_save_name'])
    p = gui.fileOpenDlg()
    p = p[0]
    print(p)
    name = p.split('/')[-1]
    l = len(name)
    path = p[0:-l]
    if infoDlg.OK == False:
        core.quit()
    # exp = data.ExperimentHandler(name='Faqce Preference',version='0.1.0')
    win = visual.Window(fullscr=True, color=(-1, -1, -1))
    w, h = win.size  # 屏幕原始像素大小
    print(w)
    print(h)
    rate = 1

    rest = visual.TextStim(win, text=u'+', pos=(0.0, 0), color='white', bold=True)
    pretext = visual.TextStim(win, text=u'提示信息', pos=(0.0, 0), color='white', bold=True)
    task1(save_path='./pos_results/', save_name=info['pos_save_name']+'.csv', load_path=path, text=u'开始实验')
    np.save(os.path.join('./time', info['pos_save_name']+'_start.npy'), time_list_s)
    np.save(os.path.join('./time', info['pos_save_name']+'_end.npy'), time_list_e)
    win.close()
    core.quit()
    udpCliSock.close()
