import os
import re
import cv2
import numpy as np
from PIL import Image


def get_all_files(directory, extension):
    jpg_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                jpg_files.append(os.path.join(root, file))

    return jpg_files

def imread(path, iscv2=False, mode='RGB'):
    # return CxHxW
    if not iscv2:
        img = (np.array(Image.open(path).convert(mode)))
    else:
        if mode == 'RGB':
            img = (
                cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        else:
            img = (
                cv2.imread(path, cv2.IMREAD_ANYDEPTH))

    return img if mode == 'RGB' else img[..., None]


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def parse_camera_file(filename):
    camera_data = {}

    with open(filename, 'r') as file:
        current_frame = None
        for line in file:
            line = line.strip()
            if line.startswith("Frame"):
                current_frame = int(line.split()[1])
                camera_data[current_frame] = {"L": [], "R": []}
            elif line.startswith("L") or line.startswith("R"):
                key = line[0]  # 'L' or 'R'
                values = list(map(float, line[2:].split()))
                camera_data[current_frame][key] = values

    return camera_data
