import os
import csv
import random
import time

from psychopy.visual import Window, TextStim, ImageStim, RatingScale
from psychopy.gui import DlgFromDict, Dlg
from psychopy.event import waitKeys, clearEvents
from psychopy.core import quit, Clock, CountdownTimer

from psychopy import parallel


time_list_s = []
time_list_e = []

port = parallel.ParallelPort(address=0x3EFC)
port.setData(0)

def trig(num):
    time_trigger_0 = time.time()
    port.setData(num)
    print('setdata time {:.10f}'.format(time.time() - time_trigger_0))
    time_list_e.append(time.time())
    dtimer = CountdownTimer(0.02)
    while dtimer.getTime() > 0:
        dddd = 1
    port.setData(0)

def wait_for_keyinput():
    clearEvents('keyboard')
    if 'escape' in waitKeys():
        win.close()
        quit()

def adjust_image_size(im, factor):
    # screen
    ## top left (-1, 1)
    ## down right (1, -1)
    ## width=2, height=2
    image_ratio = im.size[0] * 1.0 / im.size[1]
    if image_ratio > screen_ratio:
        im.size /= im.size[0]
    else:
        im.size /= im.size[1]
        
    im.size *= factor

def start_artwork_exp():
    # Welcome
    welcome_title = TextStim(win,
                             text='BCMI油画情绪分类实验',
                             font='Open Sans',
                             pos=(-0.1, 0.2),
                             height=0.2,
                             wrapWidth=None,
                             color='white',
                             colorSpace='rgb')
    welcome_text = TextStim(win,
                            text='{}, 你好，欢迎参加实验'.format(subj_info['姓名']),
                            font='Open Sans',
                            pos=(0, 0),
                            height=0.05,
                            wrapWidth=None,
                            color='white',
                            colorSpace='rgb')
    instruction_text = TextStim(win,
                                text='按任意键继续',
                                font='Open Sans',
                                pos=(0, -0.9),
                                height=0.04,
                                wrapWidth=None,
                                color='white',
                                colorSpace='rgb')
    
    welcome_title.draw()
    welcome_text.draw()
    instruction_text.draw()
    win.flip()

    # Tutorial
    tutorial_title = TextStim(win,
                              text='实验介绍',
                              font='Open Sans',
                              pos=(0, 0.8),
                              height=0.1,
                              wrapWidth=None,
                              color='white',
                              colorSpace='rgb')
    tutorial_text = TextStim(win,
                             text="油画情绪分类实验旨在探索人类应对油画艺术形式的情绪反应，在\n本次实验中，你将会看到90幅油画，请完成评分,",
                             font='Open Sans',
                             pos=(-0.9, 0.6),
                             height=0.07,
                             wrapWidth=None,
                             color='white',
                             colorSpace='rgb',
                             anchorHoriz='left',
                             anchorVert='top')
    
    tutorial_title.draw()
    tutorial_text.draw()
    instruction_text.draw()
    wait_for_keyinput()
    win.flip()
    
    # Example1
    example1_title = TextStim(win,
                              text='示例1',
                              font='Open Sans',
                              pos=(0, 0.8),
                              height=0.1,
                              wrapWidth=None,
                              color='white',
                              colorSpace='rgb')
    example_positive = ImageStim(win,
                                 image=example_pics_path+'/example_positive.jpg',
                                 pos=(0, 0))
    adjust_image_size(example_positive, 1)
    positive_tutorial = TextStim(win,
                                 text='这张画表达了正向情绪，\n在下一页中把valence滑杆和arousal滑杆滑至你认为合适的位置',
                                 font='Open Sans',
                                 pos=(0, -0.7),
                                 height=0.05,
                                 wrapWidth=None,
                                 color='white',
                                 colorSpace='rgb')

    example1_title.draw()
    example_positive.draw()
    positive_tutorial.draw()
    instruction_text.draw()
    wait_for_keyinput()
    win.flip()
    
    valence_title = TextStim(win,
                             text='看完上一幅油画，你感到开心还是悲伤？',
                             font='Open Sans',
                             pos=(0, 0.65),
                             height=0.05,
                             wrapWidth=None,
                             color='white',
                             colorSpace='rgb')
    valence_rating1 = RatingScale(win,
                                  size=1.5,
                                  textSize=0.5,
                                  low=-4,
                                  high=4,
                                  markerStart=0,
                                  precision=1,
                                  scale=None,
                                  labels=['悲伤，压抑', '没有感觉', '心情舒畅，开心'],
                                  pos=(0, 0.2),
                                  marker='circle',
                                  markerColor='DarkRed',
                                  tickHeight=1,
                                  acceptPreText='确定后,点这里',
                                  skipKeys=None,
                                  mouseOnly=True,
                                  name='valence')
    arousal_title = TextStim(win,
                             text='你的情绪有多强烈？',
                             font='Open Sans',
                             pos=(0, -0.15),
                             height=0.05,
                             wrapWidth=None,
                             color='white',
                             colorSpace='rgb')
    arousal_rating1 = RatingScale(win,
                                  size=1.5,
                                  textSize=0.5,
                                  low=-4,
                                  high=4,
                                  markerStart=0,
                                  precision=1,
                                  scale=None,
                                  labels=['不是很强烈', '适中', '非常强烈'],
                                  pos=(0, -0.6),
                                  marker='circle',
                                  markerColor='DarkGreen',
                                  tickHeight=1,
                                  acceptPreText='确定后,点这里',
                                  skipKeys=None,
                                  mouseOnly=True,
                                  name='arousal')
    valence_icons = ['./icons/v--4.jpg', './icons/v--2.jpg', './icons/v-0.jpg', './icons/v-2.jpg', './icons/v-4.jpg']
    arousal_icons = ['./icons/a--4.jpg', './icons/a--2.jpg', './icons/a-0.jpg', './icons/a-2.jpg', './icons/a-4.jpg']
    valence_imgs = []
    arousal_imgs = []
    pos = [-0.45, -0.225, 0, 0.225, 0.45]
    for i in range(5):
        vimg = ImageStim(win,
                         image=valence_icons[i],
                         pos=(pos[i], 0.45))
        # adjust_image_size(vimg, 1)
        valence_imgs.append(vimg)
        
        aimg = ImageStim(win,
                         image=arousal_icons[i],
                         pos=(pos[i], -0.35))
        # adjust_image_size(aimg, 1)
        arousal_imgs.append(aimg)

    wait_for_keyinput()
    
    while valence_rating1.noResponse or arousal_rating1.noResponse:
        example1_title.draw()
        valence_title.draw()
        arousal_title.draw()
        for vimg in valence_imgs:
            vimg.draw()
        
        for aimg in arousal_imgs:
            aimg.draw()
        
        valence_rating1.draw()
        arousal_rating1.draw()
        win.flip()
    
    example1_title.draw()
    valence_title.draw()
    arousal_title.draw()
    for vimg in valence_imgs:
        vimg.draw()
    
    for aimg in arousal_imgs:
        aimg.draw()
    
    valence_rating1.draw()
    arousal_rating1.draw()
    instruction_text.draw()
    win.flip()
    print('==1== valence rating=', valence_rating1.getRating(), ' rating time=%.3f' % valence_rating1.getRT())
    print('==1== arousal rating=', arousal_rating1.getRating(), ' rating time=%.3f' % arousal_rating1.getRT())
    
    # Example2
    example2_title = TextStim(win,
                              text='示例2',
                              font='Open Sans',
                              pos=(0, 0.8),
                              height=0.1,
                              wrapWidth=None,
                              color='white',
                              colorSpace='rgb')
    example_neutral = ImageStim(win,
                                 image=example_pics_path+'/example_neutral.jpg',
                                 pos=(0, 0))
    adjust_image_size(example_neutral, 1)
    neutral_tutorial = TextStim(win,
                                 text='这张画表达了中性情绪，\n在下一页中把valence滑杆和arousal滑杆滑至你认为合适的位置',
                                 font='Open Sans',
                                 pos=(0, -0.7),
                                 height=0.05,
                                 wrapWidth=None,
                                 color='white',
                                 colorSpace='rgb')

    example2_title.draw()
    example_neutral.draw()
    neutral_tutorial.draw()
    instruction_text.draw()
    wait_for_keyinput()
    win.flip()
    
    valence_rating2 = RatingScale(win,
                                  size=1.5,
                                  textSize=0.5,
                                  low=-4,
                                  high=4,
                                  markerStart=0,
                                  precision=1,
                                  scale=None,
                                  labels=['悲伤，压抑', '没有感觉', '心情舒畅，开心'],
                                  pos=(0, 0.2),
                                  marker='circle',
                                  markerColor='DarkRed',
                                  tickHeight=1,
                                  acceptPreText='确定后,点这里',
                                  skipKeys=None,
                                  mouseOnly=True,
                                  name='valence')
    arousal_rating2 = RatingScale(win,
                                  size=1.5,
                                  textSize=0.5,
                                  low=-4,
                                  high=4,
                                  markerStart=0,
                                  precision=1,
                                  scale=None,
                                  labels=['不是很强烈', '适中', '非常强烈'],
                                  pos=(0, -0.6),
                                  marker='circle',
                                  markerColor='DarkGreen',
                                  tickHeight=1,
                                  acceptPreText='确定后，点这里',
                                  skipKeys=None,
                                  mouseOnly=True,
                                  name='arousal')

    wait_for_keyinput()
    
    while valence_rating2.noResponse or arousal_rating2.noResponse:
        example2_title.draw()
        valence_title.draw()
        arousal_title.draw()
        for vimg in valence_imgs:
            vimg.draw()
        
        for aimg in arousal_imgs:
            aimg.draw()
        
        valence_rating2.draw()
        arousal_rating2.draw()
        win.flip()
    
    example2_title.draw()
    valence_title.draw()
    arousal_title.draw()
    for vimg in valence_imgs:
        vimg.draw()
    
    for aimg in arousal_imgs:
        aimg.draw()
    
    valence_rating2.draw()
    arousal_rating2.draw()
    instruction_text.draw()
    win.flip()
    print('==2== valence rating=', valence_rating2.getRating(), ' rating time=%.3f' % valence_rating2.getRT())
    print('==2== arousal rating=', arousal_rating2.getRating(), ' rating time=%.3f' % arousal_rating2.getRT())
    
    
    # Example3
    example3_title = TextStim(win,
                              text='示例3',
                              font='Open Sans',
                              pos=(0, 0.8),
                              height=0.1,
                              wrapWidth=None,
                              color='white',
                              colorSpace='rgb')
    example_negative = ImageStim(win,
                                 image=example_pics_path+'/example_negative.jpg',
                                 pos=(0, 0))
    adjust_image_size(example_negative, 1)
    negative_tutorial = TextStim(win,
                                 text='这张画表达了负向情绪，\n在下一页中把valence滑杆和arousal滑杆滑至你认为合适的位置',
                                 font='Open Sans',
                                 pos=(0, -0.7),
                                 height=0.05,
                                 wrapWidth=None,
                                 color='white',
                                 colorSpace='rgb')

    example3_title.draw()
    example_negative.draw()
    negative_tutorial.draw()
    instruction_text.draw()
    wait_for_keyinput()
    win.flip()
    
    valence_rating3 = RatingScale(win,
                                  size=1.5,
                                  textSize=0.5,
                                  low=-4,
                                  high=4,
                                  markerStart=0,
                                  precision=1,
                                  scale=None,
                                  labels=['悲伤，压抑', '没有感觉', '心情舒畅，开心'],
                                  pos=(0, 0.2),
                                  marker='circle',
                                  markerColor='DarkRed',
                                  tickHeight=1,
                                  acceptPreText='确定后，点这里',
                                  skipKeys=None,
                                  mouseOnly=True,
                                  name='valence')
    arousal_rating3 = RatingScale(win,
                                  size=1.5,
                                  textSize=0.5,
                                  low=-4,
                                  high=4,
                                  markerStart=0,
                                  precision=1,
                                  scale=None,
                                  labels=['不是很强烈', '适中', '非常强烈'],
                                  pos=(0, -0.6),
                                  marker='circle',
                                  markerColor='DarkGreen',
                                  tickHeight=1,
                                  acceptPreText='确定后，点这里',
                                  skipKeys=None,
                                  mouseOnly=True,
                                  name='arousal')

    wait_for_keyinput()
    
    while valence_rating3.noResponse or arousal_rating3.noResponse:
        example3_title.draw()
        valence_title.draw()
        arousal_title.draw()
        for vimg in valence_imgs:
            vimg.draw()
        
        for aimg in arousal_imgs:
            aimg.draw()
        
        valence_rating3.draw()
        arousal_rating3.draw()
        win.flip()
    
    example3_title.draw()
    valence_title.draw()
    arousal_title.draw()
    for vimg in valence_imgs:
        vimg.draw()
    
    for aimg in arousal_imgs:
        aimg.draw()
    
    valence_rating3.draw()
    arousal_rating3.draw()
    instruction_text.draw()
    win.flip()
    print('==3== valence rating=', valence_rating3.getRating(), ' rating time=%.3f' % valence_rating3.getRT())
    print('==3== arousal rating=', arousal_rating3.getRating(), ' rating time=%.3f' % arousal_rating3.getRT())
    
    
    # Main exp
    mainexp_title = TextStim(win,
                             text='下面进入正式实验',
                             font='Open Sans',
                             pos=(0, 0.2),
                             height=0.2,
                             wrapWidth=None,
                             color='white',
                             colorSpace='rgb')
    mainexp_text = TextStim(win,
                            text='{}, 请放松心情，调整呼吸'.format(subj_info['姓名']),
                            font='Open Sans',
                            pos=(0, 0),
                            height=0.05,
                            wrapWidth=None,
                            color='white',
                            colorSpace='rgb')
    mainexp_title.draw()
    mainexp_text.draw()
    instruction_text.draw()
    wait_for_keyinput()
    win.flip()
    
    pictures = [pics_path+'/'+file_name for file_name in os.listdir(pics_path)]
    total_number_of_pics = len(pictures)
    order = list(range(total_number_of_pics))
    random.shuffle(order)
    order_file = subj_dir + '/order.txt'
    with open(order_file, 'w') as of:
        of.write(str(order))
    
    progress = 1
    hint_text = TextStim(win,
                        text='进度已达一半，你可以休息一下，然后继续',
                        font='Open Sans',
                        pos=(0, 0),
                        height=0.05,
                        wrapWidth=None,
                        color='white',
                        colorSpace='rgb')
    plus = TextStim(win,
                    text='+',
                    font='Open Sans',
                    pos=(0, 0),
                    height=0.3,
                    wrapWidth=None,
                    color='white',
                    colorSpace='rgb')
    
    trial_clock = Clock()
    for p in order:
        for ppath in pictures:
            fn = int(ppath[ppath.rindex('/')+1:ppath.rindex('.')])
            if fn == p:
                paint_path = ppath
                break

        hint_title = TextStim(win,
                              text='进度：{}/{}'.format(progress, total_number_of_pics),
                              font='Open Sans',
                              pos=(-0.1, 0.2),
                              height=0.2,
                              wrapWidth=None,
                              color='white',
                              colorSpace='rgb')
        paint = ImageStim(win,
                          image=paint_path,
                          pos=(0, 0))
        adjust_image_size(paint, 2)
        
        hint_title.draw()
        if progress == total_number_of_pics // 2:
            hint_text.draw()
        instruction_text.draw()
        wait_for_keyinput()
        win.flip()
        wait_for_keyinput()
        
        pre_start_time = time.time()
        trial_clock.reset()
        trig(p)
        while trial_clock.getTime() < 20:
            if trial_clock.getTime() < 5:
                # 5s fixation
                plus.draw()
            elif trial_clock.getTime() < 15:
                # 10s presentation
                paint.draw()
            else:
                # 5s fixation
                plus.draw()
            
            win.flip()
        
        trig(p)
        pre_end_time = time.time()
        
        valence_rating = RatingScale(win,
                                  size=1.5,
                                  textSize=0.5,
                                  low=-4,
                                  high=4,
                                  markerStart=0,
                                  precision=1,
                                  scale=None,
                                  labels=['悲伤，压抑', '没有感觉', '心情舒畅，开心'],
                                  pos=(0, 0.2),
                                  marker='circle',
                                  markerColor='DarkRed',
                                  tickHeight=1,
                                  acceptPreText='确定后，点这里',
                                  skipKeys=None,
                                  mouseOnly=True,
                                  name='valence')
        arousal_rating = RatingScale(win,
                                      size=1.5,
                                      textSize=0.5,
                                      low=-4,
                                      high=4,
                                      markerStart=0,
                                      precision=1,
                                      scale=None,
                                      labels=['不是很强烈', '适中', '非常强烈'],
                                      pos=(0, -0.6),
                                      marker='circle',
                                      markerColor='DarkGreen',
                                      tickHeight=1,
                                      acceptPreText='确定后，点这里',
                                      skipKeys=None,
                                      mouseOnly=True,
                                      name='arousal')
        
        while valence_rating.noResponse or arousal_rating.noResponse:
            valence_title.draw()
            arousal_title.draw()
            for vimg in valence_imgs:
                vimg.draw()
            
            for aimg in arousal_imgs:
                aimg.draw()
            
            valence_rating.draw()
            arousal_rating.draw()
            win.flip()
        
        valence_title.draw()
        arousal_title.draw()
        for vimg in valence_imgs:
            vimg.draw()
        
        for aimg in arousal_imgs:
            aimg.draw()
        
        valence_rating.draw()
        arousal_rating.draw()
        instruction_text.draw()
        win.flip()
        
        csvwriter.writerow({fields[0]:p,
                            fields[1]:pre_start_time,
                            fields[2]:pre_end_time,
                            fields[3]:valence_rating.getRating(),
                            fields[4]:arousal_rating.getRating()})
        
        print('=={}== valence rating='.format(p), valence_rating.getRating(), ' rating time=%.3f' % valence_rating.getRT())
        print('=={}== arousal rating='.format(p), arousal_rating.getRating(), ' rating time=%.3f' % arousal_rating.getRT())
        
        progress += 1
        
    # End
    end_title = TextStim(win,
                             text='实验结束',
                             font='Open Sans',
                             pos=(-0.1, 0.2),
                             height=0.2,
                             wrapWidth=None,
                             color='white',
                             colorSpace='rgb')
    end_text = TextStim(win,
                            text='感谢参与',
                            font='Open Sans',
                            pos=(0, 0),
                            height=0.05,
                            wrapWidth=None,
                            color='white',
                            colorSpace='rgb')
    end_title.draw()
    end_text.draw()
    wait_for_keyinput()
    win.flip()
    wait_for_keyinput()
    

if __name__ == '__main__':

    pics_path = r'./pics'
    example_pics_path = r'./example_pics'
    record_dir = './artwork_exp_record'
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    # Subject information
    subj_info = {'姓名':'', '姓名拼音':'', '性别':['男', '女'], '年龄':'', '国籍':''}
    subj_info_dlg = DlgFromDict(subj_info, title='请填写个人信息', order=['姓名', '姓名拼音', '性别', '年龄', '国籍'])

    if not subj_info_dlg.OK:
        quit()
    else:
        if not subj_info['姓名'] or not subj_info['姓名拼音']:
            quit()
    
    subj_dir = record_dir + '/' + subj_info['姓名拼音']
    if not os.path.exists(subj_dir):
        os.makedirs(subj_dir)
    
    subj_file = subj_dir + '/' + subj_info['姓名拼音'] + '_info.txt'
    with open(subj_file, 'w') as sf:
        sf.write(str(subj_info))
    
    record_file = subj_dir + '/' + subj_info['姓名拼音'] + '.csv'
    csvfile = open(record_file, mode='w', encoding='utf-8', newline='')
    fields = ['image', 'pre_start_time', 'pre_end_time', 'valence_rating', 'arousal_rating']
    csvwriter = csv.DictWriter(csvfile, fieldnames=fields, delimiter=',')
    csvwriter.writeheader()
    
    # Start
    win = Window(color=(-1, -1, -1), fullscr=True, units='norm')
    screen_ratio = win.size[0] * 1.0 / win.size[1]

    start_artwork_exp()
    
    csvfile.close()
    win.close()
    quit()
    udpCliSock.close()