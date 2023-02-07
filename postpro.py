from os import listdir
import json

pose_path = '/mnt/c/Users/tkfps/Downloads/2dposes/'

def empty_frame(pose_path):
    list_2dposes = listdir(pose_path)
    total_count = 0
    position = {}
    for i in list_2dposes:   
        # print(pose_path + i)
        f = open(pose_path + i)
        temp = json.load(f)
        count = 0
        for n, j in enumerate(temp):
            if not j:
                count += 1
            if count == 1 and not position.get(i):
                position[i] = n
        if count != 0:
            print(i)
            print('empty frame: ' + str(count))
            total_count += 1
    print(total_count)
    print(position)


if __name__ == '__main__':
    empty_frame(pose_path)