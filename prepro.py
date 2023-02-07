from os import listdir
import os
import zipfile
import ffmpeg
import torch
# import cv2

dataset_path = '/mnt/d/dataset/'
rgb_path = '/mnt/c/Users/tkfps/Downloads/dataset/'
output_path = '/mnt/c/Users/tkfps/Downloads/output/dp_rgb/'
sanity_path = '/mnt/c/Users/tkfps/Downloads/b/'

# d = listdir(dataset_path)
s = listdir(sanity_path)

# check we have all zip files
def have_all(dataset, original):
    for i in original:
        temp = i.split('.')[0] + '.zip'
        # print(temp)
        if temp not in dataset:
            print(temp)

# extract rgb from zip
def get_rgb(dataset):
    for i in dataset:
        project_name = i.split('.')[0]
        if '.' in i:
            archive = zipfile.ZipFile(dataset_path+i)
            project_rgb_path = rgb_path + project_name
            project_output_path = output_path + project_name
            if not os.path.exists(project_rgb_path) and not os.path.exists(project_output_path):
                os.makedirs(project_rgb_path)
                for file in archive.namelist():
                    if file.startswith('rgb/'):
                        archive.extract(file, project_rgb_path)

def sanity_check(dataset):
    empty = []
    for i in dataset:
        # print(i)
        subfolder = listdir(rgb_path+i)
        # print(subfolder)
        if not subfolder:
            empty.append(i)
    return empty
        
def get_bitrate(file):
    try:
        probe = ffmpeg.probe(file)
        video_bitrate = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        bitrate = int(int(video_bitrate['bit_rate']) / 1000)
        return bitrate
    except:
        print('wut')

def get_framerate(file):
    probe = ffmpeg.probe(file)
    return probe['streams'][0]['r_frame_rate'].split('/')[0]


def img_to_video(img_path, video_path, out_path):
    '''
    video_path: to get frame rate and bitrate
    '''
    img_arg = img_path + '/%010d.png'
    frame_rate = get_framerate(video_path)
    bitrate = str(get_bitrate(video_path))+'k'
    arg = 'ffmpeg -r ' + frame_rate + ' -i ' + img_arg + ' -vcodec mpeg4 -b:v ' + bitrate + ' -y ' + out_path
    os.system(arg)


# get fps
# cap = cv2.VideoCapture("insert path here")
# fps = cap.get(cv2.CAP_PROP_FPS)

if __name__ == '__main__':
    # print()
    # empty_rgb = sanity_check(listdir(rgb_path))
    # print(empty_rgb)
    # missed = []
    # for i in s:
        
    #     temp = i.split('.')[0]
    #     if temp not in listdir(output_path):
    #         missed.append(temp)
    # print(missed)
    # print(len(missed))
    # have_all(d, s)
    # get_bitrate('/mnt/c/Users/tkfps/Downloads/b/20220812_125132.mp4')
    for i in s:
        # vid_file = sanity_path + i
        # img_dir = '/mnt/c/Users/tkfps/Downloads/output/dp_rgb/' + i.split('.')[0] + '/rgb/'
        # vid_out = '/mnt/c/Users/tkfps/Downloads/output/dp_vid/' + i
        # img_to_video(img_dir, vid_file, vid_out)
        
        b_probe = ffmpeg.probe('/mnt/c/Users/tkfps/Downloads/b/' + i)
        a_probe = ffmpeg.probe('/mnt/c/Users/tkfps/Downloads/output/dp_vid/' + i)
        b_video_stream = next((stream for stream in b_probe['streams'] if stream['codec_type'] == 'video'), None)['nb_frames']
        a_video_stream = next((stream for stream in a_probe['streams'] if stream['codec_type'] == 'video'), None)['nb_frames']

        if b_video_stream != a_video_stream:
            print('noooo')
            print(i)
            print(b_video_stream)
            print(a_video_stream)
        
    #  ffmpeg -r 15 -i /mnt/c/Users/tkfps/Downloads/output/dp_rgb/20210518_230219/%010d.png -vcodec mpeg4 -b:v 581k -y /mnt/c/Users/tkfps/Downloads/output/20210518_230219.mp4