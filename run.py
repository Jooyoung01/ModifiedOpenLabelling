import argparse
import textwrap
import glob
import os
import shutil

import numpy as np
import cv2

bin_bbox_path = 'bin/bbox_txt'
bin_images_path = 'bin/images'
new_label_path = 'new_label'

if not os.path.isdir(bin_bbox_path):
    os.makedirs(bin_bbox_path)
if not os.path.isdir(bin_images_path):
    os.makedirs(bin_images_path)
if not os.path.isdir(new_label_path):
    os.makedirs(new_label_path)

WITH_QT = True
try:
    cv2.namedWindow("Test")
    cv2.displayOverlay("Test", "Test QT", 1000)
except:
    WITH_QT = False
cv2.destroyAllWindows()

bbox_thickness = 1

parser = argparse.ArgumentParser(description='YOLO Bounding Box Tool')
parser.add_argument('--format', default='yolo', type=str, choices=['yolo', 'voc'], help="Bounding box format")
parser.add_argument('--sort', action='store_true', help="If true, shows images in order.")
parser.add_argument('--cross-thickness', default='1', type=int, help="Cross thickness")
parser.add_argument('--bbox-thickness', default=bbox_thickness, type=int, help="Bounding box thickness")
args = parser.parse_args()

class_index = 0
img_index = 0
img = None
channels = []
channel_index = 0
img_objects = []
bb_dir = "bbox_txt/"

# selected bounding box
prev_was_double_click = False
is_bbox_selected = False
selected_bbox = -1

mouse_x = 0
mouse_y = 0
point_1 = (-1, -1)
point_2 = (-1, -1)

show_labels = True  # label Flag
edges_on = False  # edges Flag

# Information window setup
INFO_WINDOW = 'Info Window'
cv2.namedWindow(INFO_WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(INFO_WINDOW, 400, 300)

def update_info_window():
    info_img = np.zeros((300, 400, 3), dtype=np.uint8)
    filename = os.path.basename(image_list[img_index])
    total_labels = len(img_objects)
    img_height, img_width = img.shape[:2]
    help_text = "[h] for help\n[a/d] to change image\n[w/s] to change class\n[l] to toggle labels\n[b] to cycle channels"
    y0, dy = 20, 20

    channel_name = ["BGR", "Blue", "Green", "Red", "NIR"][channel_index] if channel_index < 5 else f"Channel {channel_index}"

    for i, line in enumerate(help_text.split('\n')):
        y = y0 + i * dy
        cv2.putText(info_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(info_img, f"File: {filename}", (10, y0 + (len(help_text.split('\n')) + 1) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(info_img, f"Total Labels: {total_labels}", (10, y0 + (len(help_text.split('\n')) + 2) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(info_img, f"Channel: {channel_name}", (10, y0 + (len(help_text.split('\n')) + 3) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(info_img, f"Image Size: {img_width}x{img_height}", (10, y0 + (len(help_text.split('\n')) + 4) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(info_img, f"Labels: {'ON' if show_labels else 'OFF'}", (10, y0 + (len(help_text.split('\n')) + 5) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(info_img, f"Edges: {'ON' if edges_on else 'OFF'}", (10, y0 + (len(help_text.split('\n')) + 6) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(INFO_WINDOW, info_img)

def change_img_index(x):
    global img_index, img, channels, channel_index
    img_index = x
    img_path = image_list[img_index]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    channels = create_color_channels(img)
    channel_index = 0
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, "Showing image "
                                    "" + str(img_index) + "/"
                                    "" + str(last_img_index), 1000)
    else:
        print("Showing image "
                "" + str(img_index) + "/"
                "" + str(last_img_index) + " path:" + img_path)
    update_info_window()

def create_color_channels(img):
    channels = [img]
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        channels += [cv2.merge([b, np.zeros_like(b), np.zeros_like(b)]),
                     cv2.merge([np.zeros_like(g), g, np.zeros_like(g)]),
                     cv2.merge([np.zeros_like(r), np.zeros_like(r), r])]
    elif img.shape[2] == 4:
        b, g, r, nir = cv2.split(img)
        channels += [cv2.merge([b, np.zeros_like(b), np.zeros_like(b)]),
                     cv2.merge([np.zeros_like(g), g, np.zeros_like(g)]),
                     cv2.merge([np.zeros_like(r), np.zeros_like(r), r]),
                     cv2.merge([nir, nir, nir])]
    return channels

def change_class_index(x):
    global class_index
    class_index = x
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, "Selected class "
                                "" + str(class_index) + "/"
                                "" + str(last_class_index) + ""
                                "\n " + class_list[class_index],3000)
    else:
        print("Selected class :" + class_list[class_index])
    update_info_window()

def draw_edges(tmp_img):
    if tmp_img.ndim == 2:
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_GRAY2BGR)
    elif tmp_img.shape[2] > 3:
        tmp_img = tmp_img[:, :, :3]
    blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
    edges = cv2.Canny(blur, 150, 250, 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Overlap image and edges together
    tmp_img = np.bitwise_or(tmp_img, edges)
    return tmp_img

def decrease_index(current_index, last_index):
    current_index -= 1
    if current_index < 0:
        current_index = last_index
    return current_index

def increase_index(current_index, last_index):
    current_index += 1
    if current_index > last_index:
        current_index = 0
    return current_index

def draw_line(img, x, y, height, width, color):
    cv2.line(img, (x, 0), (x, height), color, thickness=args.cross_thickness)
    cv2.line(img, (0, y), (width, y), color, thickness=args.cross_thickness)

def yolo_format(class_index, point_1, point_2, width, height):
    x_center = (point_1[0] + point_2[0]) / float(2.0 * width)
    y_center = (point_1[1] + point_2[1]) / float(2.0 * height)
    x_width = float(abs(point_2[0] - point_1[0])) / width
    y_height = float(abs(point_2[1] - point_1[1])) / height
    return str(class_index) + " " + str(x_center) \
        + " " + str(y_center) + " " + str(x_width) + " " + str(y_height)

def voc_format(class_index, point_1, point_2):
    xmin, ymin = min(point_1[0], point_2[0]) + 1, min(point_1[1], point_2[1]) + 1
    xmax, ymax = max(point_1[0], point_2[0]) + 1, max(point_1[1], point_2[1]) + 1
    items = map(str, [xmin, ymin, xmax, ymax, class_index])
    return ' '.join(items)

def get_txt_path(img_path):
    img_name = os.path.basename(os.path.normpath(img_path))
    img_type = img_path.split('.')[-1]
    new_txt_path = os.path.join(new_label_path, img_name.replace(img_type, 'txt'))
    if os.path.exists(new_txt_path):
        return new_txt_path
    return os.path.join(bb_dir, img_name.replace(img_type, 'txt'))

def copy_to_new_label(txt_path):
    new_txt_path = os.path.join(new_label_path, os.path.basename(txt_path))
    if not os.path.exists(new_txt_path):
        shutil.copy(txt_path, new_txt_path)
    return new_txt_path

def save_bb(txt_path, line):
    new_txt_path = copy_to_new_label(txt_path)
    with open(new_txt_path, 'a') as myfile:
        myfile.write(line + "\n")
    update_info_window()

def delete_bb(txt_path, line_index):
    new_txt_path = copy_to_new_label(txt_path)
    with open(new_txt_path, "r") as old_file:
        lines = old_file.readlines()

    with open(new_txt_path, "w") as new_file:
        counter = 0
        for line in lines:
            if counter != line_index:
                new_file.write(line)
            counter += 1
    update_info_window()

def yolo_to_x_y(x_center, y_center, x_width, y_height, width, height):
    x_center *= width
    y_center *= height
    x_width *= width
    y_height *= height
    x_width /= 2.0
    y_height /= 2.0
    return int(x_center - x_width), int(y_center - y_height), int(x_center + x_width), int(y_center + y_height)

def draw_text(tmp_img, text, center, color, size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(tmp_img, text, center, font, 0.7, color, size, cv2.FONT_HERSHEY_COMPLEX_SMALL)
    return tmp_img

def draw_bboxes_from_file(tmp_img, txt_path, width, height):
    global img_objects
    img_objects = []
    if os.path.isfile(txt_path):
        with open(txt_path) as f:
            content = f.readlines()
        for line in content:
            values_str = line.split()
            # print(f"Processing line: {values_str}")  # Debug print to see the values being read
            if args.format == 'yolo':
                if len(values_str) == 5:
                    class_index, x_center, y_center, x_width, y_height = map(float, values_str)
                    confid = 1.0  # confidence 1.0
                elif len(values_str) == 6:
                    class_index, x_center, y_center, x_width, y_height, confid = map(float, values_str)
                else:
                    error = ("You selected the 'yolo' format but your labels "
                            "seem to be in a different format. Consider "
                            "removing your old label files.")
                    raise Exception(textwrap.fill(error, 70))
                class_index = int(class_index)
                x1, y1, x2, y2 = yolo_to_x_y(x_center, y_center, x_width, y_height, width, height)
                img_objects.append([class_index, x1, y1, x2, y2])
                color = class_rgb[class_index].tolist()
                cv2.rectangle(tmp_img, (x1, y1), (x2, y2), color, thickness=args.bbox_thickness)
                if show_labels:  # show_labels flag
                    tmp_img = draw_text(tmp_img, class_list[class_index] + '_' + str(confid), (x1, y1 - 5), color, args.bbox_thickness)
            elif args.format == 'voc':
                try:
                    x1, y1, x2, y2, class_index = map(int, values_str)
                except ValueError:
                    error = ("You selected the 'voc' format but your labels "
                            "seem to be in a different format. Consider "
                            "removing your old label files.")
                    raise Exception(textwrap.fill(error, 70))
                x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 - 1, y2 - 1
                img_objects.append([class_index, x1, y1, x2, y2])
                color = class_rgb[class_index].tolist()
                cv2.rectangle(tmp_img, (x1, y1), (x2, y2), color, thickness=args.bbox_thickness)
                if show_labels:  # show_labels flag
                    tmp_img = draw_text(tmp_img, class_list[class_index] + '_' + str(confid), (x1, y1 - 5), color, args.bbox_thickness)
    return tmp_img

def get_bbox_area(x1, y1, x2, y2):
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width * height

def set_selected_bbox():
    global is_bbox_selected, selected_bbox
    smallest_area = -1
    for idx, obj in enumerate(img_objects):
        ind, x1, y1, x2, y2 = obj
        if is_mouse_inside_points(x1, y1, x2, y2):
            is_bbox_selected = True
            tmp_area = get_bbox_area(x1, y1, x2, y2)
            if tmp_area < smallest_area or smallest_area == -1:
                smallest_area = tmp_area
                selected_bbox = idx
    update_info_window()

def mouse_inside_delete_button():
    for idx, obj in enumerate(img_objects):
        if idx == selected_bbox:
            ind, x1, y1, x2, y2 = obj
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            if is_mouse_inside_points(x1_c, y1_c, x2_c, y2_c):
                return True
    return False

def delete_selected_bbox():
    img_path = image_list[img_index]
    txt_path = get_txt_path(img_path)
    is_bbox_selected = False

    with open(txt_path, "r") as old_file:
        lines = old_file.readlines()

    with open(txt_path, "w") as new_file:
        counter = 0
        for line in lines:
            if counter != selected_bbox:
                new_file.write(line)
            counter += 1
    update_info_window()

# mouse callback function
def mouse_listener(event, x, y, flags, param):
    global is_bbox_selected, prev_was_double_click, mouse_x, mouse_y, point_1, point_2

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        prev_was_double_click = True
        point_1 = (-1, -1)
        set_selected_bbox()
    elif event == cv2.EVENT_RBUTTONDOWN:
        set_selected_bbox()
        if is_bbox_selected:
            delete_selected_bbox()
    elif event == cv2.EVENT_LBUTTONDOWN:
        if prev_was_double_click:
            prev_was_double_click = False

        is_mouse_inside_delete_button = mouse_inside_delete_button()
        if point_1[0] == -1:
            if is_bbox_selected and is_mouse_inside_delete_button:
                delete_selected_bbox()
            else:
                is_bbox_selected = False
                point_1 = (x, y)
        else:
            threshold = 5
            if abs(x - point_1[0]) > threshold or abs(y - point_1[1]) > threshold:
                point_2 = (x, y)
        update_info_window()

def is_mouse_inside_points(x1, y1, x2, y2):
    return mouse_x > x1 and mouse_x < x2 and mouse_y > y1 and mouse_y < y2

def get_close_icon(x1, y1, x2, y2):
    percentage = 0.05
    height = -1
    while height < 15 and percentage < 1.0:
        height = int((y2 - y1) * percentage)
        percentage += 0.1
    return (x2 - height), y1, x2, (y1 + height)

def draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c):
    red = (0, 0, 255)
    cv2.rectangle(tmp_img, (x1_c + 1, y1_c - 1), (x2_c, y2_c), red, -1)
    white = (255, 255, 255)
    cv2.line(tmp_img, (x1_c, y1_c), (x2_c, y2_c), white, 2)
    cv2.line(tmp_img, (x1_c, y2_c), (x2_c, y1_c), white, 2)
    return tmp_img

def draw_info_bb_selected(tmp_img):
    for idx, obj in enumerate(img_objects):
        ind, x1, y1, x2, y2 = obj
        if idx == selected_bbox:
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c)
    return tmp_img

def remove_bad_data(img_path, img_path_txt):
    img_name = img_path.split('/')[-1]
    txt_name = img_path_txt.split('/')[-1]

    os.rename(img_path, os.path.join('bin/images', img_name))
    os.rename(img_path_txt, os.path.join('bin/bbox_txt', txt_name))

img_dir = "images/"
image_list = []
for f in os.listdir(img_dir):
    f_path = os.path.join(img_dir, f)
    test_img = cv2.imread(f_path)
    if test_img is not None:
        image_list.append(f_path)

last_img_index = len(image_list) - 1
print(image_list)

if not os.path.exists(bb_dir):
    os.makedirs(bb_dir)

for img_path in image_list:
    txt_path = get_txt_path(img_path)
    if not os.path.isfile(txt_path):
        open(txt_path, 'a').close()

with open('class_list.txt') as f:
    class_list = f.read().splitlines()
last_class_index = len(class_list) - 1

class_rgb = [
    (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
    (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)]
class_rgb = np.array(class_rgb)
num_colors_missing = len(class_list) - len(class_rgb)
if num_colors_missing > 0:
    more_colors = np.random.randint(0, 255 + 1, size=(num_colors_missing, 3))
    class_rgb = np.vstack([class_rgb, more_colors])

WINDOW_NAME = 'Bounding Box Labeler'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME, 500, 500)
cv2.setMouseCallback(WINDOW_NAME, mouse_listener)

TRACKBAR_IMG = 'Image'
cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, 0, last_img_index, change_img_index)

TRACKBAR_CLASS = 'Class'
if last_class_index != 0:
    cv2.createTrackbar(TRACKBAR_CLASS, WINDOW_NAME, 0, last_class_index, change_class_index)

change_img_index(0)

if WITH_QT:
    cv2.displayOverlay(WINDOW_NAME, "Welcome!\n Press [h] for help.", 4000)
print(" Welcome!\n Select the window and press [h] for help.")

color = class_rgb[class_index].tolist()
while True:
    tmp_img = channels[channel_index].copy()
    height, width = tmp_img.shape[:2]
    if edges_on:
        try:
            tmp_img = draw_edges(tmp_img)
        except ValueError as e:
            print(f"Error applying edges: {e}")

    img_path = image_list[img_index]
    txt_path = get_txt_path(img_path)

    try:
        tmp_img = draw_bboxes_from_file(tmp_img, txt_path, width, height)
    except Exception as e:
        print(f"Error processing file {txt_path}: {e}")
        continue

    if is_bbox_selected:
        tmp_img = draw_info_bb_selected(tmp_img)
    if point_1[0] != -1:
        color = class_rgb[class_index].tolist()
        cv2.rectangle(tmp_img, point_1, (mouse_x, mouse_y), color, thickness=args.bbox_thickness)
        if point_2[0] != -1:
            if args.format == 'yolo':
                line = yolo_format(class_index, point_1, point_2, width, height)
            elif args.format == 'voc':
                line = voc_format(class_index, point_1, point_2)
            save_bb(txt_path, line)
            point_1 = (-1, -1)
            point_2 = (-1, -1)
        else:
            if WITH_QT:
                cv2.displayOverlay(WINDOW_NAME, "Selected label: " + class_list[class_index] + ""
                                    "\nPress [w] or [s] to change.", 120)

    cv2.imshow(WINDOW_NAME, tmp_img)
    update_info_window()
    pressed_key = cv2.waitKey(50)

    """ Key Listeners START """
    if pressed_key == ord('a') or pressed_key == ord('d'):
        if pressed_key == ord('a'):
            img_index = decrease_index(img_index, last_img_index)
        elif pressed_key == ord('d'):
            img_index = increase_index(img_index, last_img_index)
        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)

    elif pressed_key == ord('s') or pressed_key == ord('w'):
        if pressed_key == ord('s'):
            class_index = decrease_index(class_index, last_class_index)
        elif pressed_key == ord('w'):
            class_index = increase_index(class_index, last_class_index)
        color = class_rgb[class_index].tolist()
        draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
        cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)

    elif pressed_key == ord('r'):
        bad_path = img_path
        bad_text = txt_path

        img_index = increase_index(img_index, last_img_index)
        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)

        if img_index == 0:
            del image_list[last_img_index]
            last_img_index = len(image_list) - 1

            remove_bad_data(bad_path, bad_text)

            img_index -= 0

        else:
            del image_list[img_index - 1]
            last_img_index = len(image_list) - 1

            remove_bad_data(bad_path, bad_text)

            img_index -= 1

        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)

    elif pressed_key == ord('1'):
        if len(class_list) >= 1:
            class_index = 0
            color = class_rgb[class_index].tolist()
            draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
            cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)

    elif pressed_key == ord('2'):
        if len(class_list) >= 2:
            class_index = 1
            color = class_rgb[class_index].tolist()
            draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
            cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)

    elif pressed_key == ord('3'):
        if len(class_list) >= 3:
            class_index = 2
            color = class_rgb[class_index].tolist()
            draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
            cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)

    elif pressed_key == ord('4'):
        if len(class_list) >= 4:
            class_index = 3
            color = class_rgb[class_index].tolist()
            draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
            cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)

    elif pressed_key == ord('5'):
        if len(class_list) >= 5:
            class_index = 4
            color = class_rgb[class_index].tolist()
            draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
            cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)

    elif pressed_key == ord('6'):
        if len(class_list) >= 6:
            class_index = 5
            color = class_rgb[class_index].tolist()
            draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
            cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)

    elif pressed_key == ord('7'):
        if len(class_list) >= 7:
            class_index = 6
            color = class_rgb[class_index].tolist()
            draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
            cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)

    elif pressed_key == ord('8'):
        if len(class_list) >= 8:
            class_index = 7
            color = class_rgb[class_index].tolist()
            draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
            cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)

    elif pressed_key == ord('9'):
        if len(class_list) >= 9:
            class_index = 8
            color = class_rgb[class_index].tolist()
            draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
            cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)

    elif pressed_key == ord('h'):
        if WITH_QT:
            cv2.displayOverlay(WINDOW_NAME, "[e] to show edges;\n"
                                "[q] to quit;\n"
                                "[a] or [d] to change Image;\n"
                                "[w] or [s] to change Class.\n"
                                "[l] to toggle labels on/off.\n"
                                "[b] to cycle channels.\n"
                                "%s" % img_path, 6000)
        else:
            print("[e] to show edges;\n"
                    "[q] to quit;\n"
                    "[a] or [d] to change Image;\n"
                    "[w] or [s] to change Class.\n"
                    "[l] to toggle labels on/off.\n"
                    "[b] to cycle channels.\n"
                    "%s" % img_path)
    elif pressed_key == ord('e'):
        edges_on = not edges_on
        if WITH_QT:
            cv2.displayOverlay(WINDOW_NAME, "Edges turned {}".format("ON" if edges_on else "OFF"), 1000)
        else:
            print("Edges turned {}".format("ON" if edges_on else "OFF"))
        update_info_window()

    elif pressed_key == ord('q'):
        break

    elif pressed_key == ord('l'):  # 'l' key show_labels to toggle label flag
        show_labels = not show_labels
        if WITH_QT:
            cv2.displayOverlay(WINDOW_NAME, "Labels {}".format("ON" if show_labels else "OFF"), 1000)
        else:
            print("Labels {}".format("ON" if show_labels else "OFF"))
        update_info_window()

    elif pressed_key == ord('b'):  # 'b' key to cycle through channels
        channel_index = (channel_index + 1) % len(channels)
        if WITH_QT:
            cv2.displayOverlay(WINDOW_NAME, "Channel {}".format(channel_index), 1000)
        else:
            print("Channel {}".format(channel_index))
        update_info_window()

    """ Key Listeners END """

    if WITH_QT:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

cv2.destroyAllWindows()
