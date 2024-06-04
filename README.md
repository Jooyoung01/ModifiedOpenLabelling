# Bounding Box Labeler Documentation

## Overview
Bounding Box Labeler is a tool for labeling objects in images and drawing bounding boxes. This software supports **YOLO** and VOC formats for bounding boxes and uses OpenCV to visualize and edit images and labels. It is useful for training machine learning models on labeled images.

## Installation and Execution
### Requirements
- Python 3.x
- OpenCV
- numpy

### Installation
1. Ensure Python is installed.
2. Install the required libraries:
    ```bash
    pip install numpy opencv-python
    ```

### Execution
1. Download or clone the software code.
2. Open a terminal or command prompt and navigate to the directory containing the software.
3. Run the following command to start the Bounding Box Labeler:
    ```bash
    python labeler.py --format yolo --sort --cross-thickness 1 --bbox-thickness 1
    ```

## Usage
### Basic Interface
When you run the Bounding Box Labeler, a window will appear where you can view images and draw bounding boxes. The basic interface includes:
- Image Window: Displays the image and allows you to draw bounding boxes.
- Trackbars: Allows you to select images and classes.

### Mouse Controls
- Left Click: Select the first point of the bounding box. Select the second point with another click to draw the box.
- Right Click: Delete the selected bounding box.
- Double Click: Select a bounding box.

### Keyboard Controls
- `a` or `d`: Move to the previous or next image.
- `w` or `s`: Move to the previous or next class.
- `r`: Delete the current image and move to the next image.
- Number keys (`1` - `9`): Move to the corresponding class index.
- `h`: Display help.
- `e`: Toggle edge detection.
- `q`: Quit the program.
- `l`: Toggle label display.
- `b`: Cycle through channels.

## Code Explanation
### Key Functions
- `change_img_index(x)`: Change the image index and load the image.
- `change_class_index(x)`: Change the class index.
- `draw_edges(tmp_img)`: Draw edges on the image.
- `draw_line(img, x, y, height, width, color)`: Draw a line on the image.
- `yolo_format(class_index, point_1, point_2, width, height)`: Return the bounding box in YOLO format.
- `voc_format(class_index, point_1, point_2)`: Return the bounding box in VOC format.
- `get_txt_path(img_path)`: Return the path to the text file corresponding to the image path.
- `copy_to_new_label(txt_path)`: Copy the original label file to the `new_label` folder if a modification occurs.
- `save_bb(txt_path, line)`: Save the bounding box.
- `delete_bb(txt_path, line_index)`: Delete the bounding box.
- `yolo_to_x_y(x_center, y_center, x_width, y_height, width, height)`: Convert YOLO coordinates to x, y coordinates.
- `draw_text(tmp_img, text, center, color, size)`: Draw text on the image.
- `draw_bboxes_from_file(tmp_img, txt_path, width, height)`: Read bounding boxes from a file and draw them on the image.

## Notes
- Modified labels are saved in the `new_label` folder while maintaining the original files.
- If a label file does not exist in the `new_label` folder, it is read from the `bbox_txt` folder.

## License
Bounding Box Labeler is distributed under the MIT License. See the LICENSE file for more information.

---

This document provides an overview of the functionality and usage of the Bounding Box Labeler. If you have any questions or issues, please report them on the issue tracker.


--- 
The previous Readme file:

# ModifiedOpenLabelling
A modified version of https://github.com/Cartucho/OpenLabeling OpenLabelling tool.

This repo is used as an example labelling tool in the [YOLOv5 Series](https://www.youtube.com/playlist?list=PLD80i8An1OEHEpJVjtujEb0lQWc0GhX_4).

![](https://user-images.githubusercontent.com/41416855/122698979-26331d00-d251-11eb-8d02-f4b479e8c0df.png)


