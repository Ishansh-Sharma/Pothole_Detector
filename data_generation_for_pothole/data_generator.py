import os
import json
import cv2
import numpy as np
import random
from tqdm import tqdm

def create_oval_mask(image_shape, center, axes, angle=0):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    return mask

def add_pothole_oval(image):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    lower_third_start = int(height * 2 / 3)
    
    center_x = random.randint(width // 4, 3 * width // 4)
    center_y = random.randint(lower_third_start, height - 50)
    axes_x = random.randint(40, 60)
    axes_y = random.randint(20, 30)
    angle = random.randint(-15, 15)
    
    oval_mask = create_oval_mask(image.shape, (center_x, center_y), (axes_x, axes_y), angle)
    mask = cv2.bitwise_or(mask, oval_mask)
    image[oval_mask > 0] = [255, 255, 255]
    
    oval_info = {
        'center': (center_x, center_y),
        'axes': (axes_x, axes_y),
        'angle': angle
    }
    
    return image, mask, oval_info

def process_dataset(split_name, image_dir, mask_dir, annotation_path, output_image_dir, output_mask_dir, output_annotation_path):
    print(f"Processing {split_name} dataset...")

    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    annotation_id = max([ann['id'] for ann in coco_data['annotations']], default=0) + 1

    for img_info in tqdm(coco_data['images']):
        img_path = os.path.join(image_dir, img_info['file_name'])
        mask_path = os.path.join(mask_dir, img_info['file_name'].replace('.jpg', '.png'))

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"Warning: Missing files for {img_info['file_name']}, skipping...")
            continue

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Warning: Could not load images for {img_info['file_name']}, skipping...")
            continue

        image_with_oval, oval_mask, oval = add_pothole_oval(image.copy())

        multi_class_mask = np.zeros_like(mask)
        lane_mask = (mask > 0).astype(np.uint8) * 1
        oval_mask_binary = (oval_mask > 0).astype(np.uint8) * 2
        multi_class_mask = np.maximum(multi_class_mask, lane_mask)
        multi_class_mask = np.maximum(multi_class_mask, oval_mask_binary)

        new_img_path = os.path.join(output_image_dir, img_info['file_name'])
        new_mask_path = os.path.join(output_mask_dir, img_info['file_name'].replace('.jpg', '.png'))

        cv2.imwrite(new_img_path, image_with_oval)
        cv2.imwrite(new_mask_path, multi_class_mask)

        center_x, center_y = oval['center']
        axes_x, axes_y = oval['axes']
        angle = oval['angle']

        ellipse_points = []
        for t in range(0, 360, 5):
            rad = np.radians(t)
            x = center_x + axes_x * np.cos(rad)
            y = center_y + axes_y * np.sin(rad)
            ellipse_points.extend([float(x), float(y)])

        new_annotation = {
            'id': annotation_id,
            'image_id': img_info['id'],
            'category_id': 2,
            'segmentation': [ellipse_points],
            'area': float(np.pi * axes_x * axes_y),
            'bbox': [
                float(center_x - axes_x),
                float(center_y - axes_y),
                float(2 * axes_x),
                float(2 * axes_y)
            ],
            'iscrowd': 0
        }
        coco_data['annotations'].append(new_annotation)
        annotation_id += 1

    if not any(cat['id'] == 2 for cat in coco_data['categories']):
        coco_data['categories'].append({
            'id': 2,
            'name': 'oval',
            'supercategory': 'shape'
        })

    with open(output_annotation_path, 'w') as f:
        json.dump(coco_data, f, indent=2, default=str)

if __name__ == '__main__':
    dataset_paths = {
        'train': {
            'image_dir': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/train/images',
            'mask_dir': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/train/masks',
            'annotation_path': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/train/_annotations.coco.json',
            'output_image_dir': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/train/images_with_ovals',
            'output_mask_dir': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/train/masks_with_ovals',
            'output_annotation_path': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/train/_annotations_with_ovals.coco.json'
        },
        'valid': {
            'image_dir': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/valid/images',
            'mask_dir': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/valid/masks',
            'annotation_path': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/valid/_annotations.coco.json',
            'output_image_dir': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/valid/images_with_ovals',
            'output_mask_dir': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/valid/masks_with_ovals',
            'output_annotation_path': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/valid/_annotations_with_ovals.coco.json'
        },
        'test': {
            'image_dir': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/test/images',
            'mask_dir': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/test/masks',
            'annotation_path': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/test/_annotations.coco.json',
            'output_image_dir': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/test/images_with_ovals',
            'output_mask_dir': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/test/masks_with_ovals',
            'output_annotation_path': '/Users/ishanshsharma/Desktop/Pothole Detection/Lane_Detection_dataset/test/_annotations_with_ovals.coco.json'
        }
    }

    for split, paths in dataset_paths.items():
        process_dataset(
            split,
            paths['image_dir'],
            paths['mask_dir'],
            paths['annotation_path'],
            paths['output_image_dir'],
            paths['output_mask_dir'],
            paths['output_annotation_path']
        )


