import cv2
import os

output_folder_name = 'export'
if not os.path.exists(output_folder_name):
  os.makedirs(output_folder_name)
video_path = '270_2pm_0000.122_0061.110.mp4'
output_path = rf'{output_folder_name}/frame_%06d.jpg'
cap = cv2.VideoCapture(video_path)
frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite(output_path % frame_id, frame)
    frame_id += 1

cap.release()
