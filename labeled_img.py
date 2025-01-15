import os
import cv2
from tqdm import tqdm
import argparse
import re

def human_sort(files):
    """Sort file names in a human-readable order."""
    return sorted(files, key=lambda x: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', x)])

def draw_bounding_boxes(image_folder, label_folder, output_folder, visualize_only):
    # Ensure output folder exists if not in visualize_only mode
    if not visualize_only and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if visualize_only:
        # Visualize images in labeled_output
        labeled_images = human_sort([f for f in os.listdir(output_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        if not labeled_images:
            print(f"No images found in {output_folder} for visualization.")
            return

        index = 0

        def update_image(val):
            nonlocal index
            index = val
            display_image(index)

        def display_image(index):
            image_file = labeled_images[index]
            image_path = os.path.join(output_folder, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to read image: {image_path}")
                return

            # Create a blank canvas for file name display
            canvas = cv2.copyMakeBorder(image, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            file_name = f"File: {image_file}"
            cv2.putText(canvas, file_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Image", canvas)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.createTrackbar("Image Index", "Image", 0, len(labeled_images) - 1, update_image)

        display_image(index)

        while True:
            key = cv2.waitKey(0)
            if key == 27 or key == ord('q'):  # ESC or q key to exit visualization
                break
            elif key == ord('d') and index < len(labeled_images) - 1:  # Next image
                index += 1
                cv2.setTrackbarPos("Image Index", "Image", index)
                update_image(index)
            elif key == ord('a') and index > 0:  # Previous image
                index -= 1
                cv2.setTrackbarPos("Image Index", "Image", index)
                update_image(index)

        cv2.destroyAllWindows()
    else:
        # Get list of images and labels
        image_files = human_sort([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        label_files = [f for f in os.listdir(label_folder) if f.lower().endswith('.txt')]

        # Match labels to images
        matched_labels = 0
        for image_file in image_files:
            label_file = os.path.join(label_folder, os.path.splitext(image_file)[0] + ".txt")
            if os.path.exists(label_file):
                matched_labels += 1

        print(f"Total images: {len(image_files)}")
        print(f"Total labels: {len(label_files)}")
        print(f"Labels matched with images: {matched_labels}")

        for image_file in tqdm(image_files, desc="Processing images"):
            label_file = os.path.join(label_folder, os.path.splitext(image_file)[0] + ".txt")

            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            h, w, _ = image.shape

            if os.path.exists(label_file):
                with open(label_file, 'r') as file:
                    lines = file.readlines()

                for line in lines:
                    data = line.strip().split()
                    if len(data) != 5:
                        print(f"Skipping invalid label in {label_file}: {line}")
                        continue

                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, data)

                    x_center *= w
                    y_center *= h
                    bbox_width *= w
                    bbox_height *= h

                    x1 = int(x_center - bbox_width / 2)
                    y1 = int(y_center - bbox_height / 2)
                    x2 = int(x_center + bbox_width / 2)
                    y2 = int(y_center + bbox_height / 2)

                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w - 1, x2))
                    y2 = max(0, min(h - 1, y2))

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"Class {int(class_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, image)

        print(f"Labeled images saved in {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw YOLO bounding boxes on images.")
    parser.add_argument("--img_path", type=str, default="images", help="Path to the folder containing images.")
    parser.add_argument("--label_path", type=str, default="new_label", help="Path to the folder containing label files.")
    parser.add_argument("--output_path", type=str, default="labeled_output", help="Path to save the labeled images.")
    parser.add_argument("--vis_only", action="store_true", help="Only visualize the images in labeled_output.")
    
    args = parser.parse_args()

    draw_bounding_boxes(args.img_path, args.label_path, args.output_path, args.vis_only)
