import os
import shutil
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split

def explore_dataset(dataset_path):
    """Explore the dataset structure"""
    print("Exploring dataset structure...")
    dataset_path = Path(dataset_path)
    
    # Common ASL dataset structures
    structures = {
        'images': [],
        'labels': [],
        'classes': []
    }
    
    for item in dataset_path.rglob('*'):
        if item.is_file():
            if item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                structures['images'].append(item)
            elif item.suffix.lower() in ['.txt', '.xml']:
                structures['labels'].append(item)
    
    # Check for class folders (classification format)
    class_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    if class_folders:
        print(f"Found {len(class_folders)} class folders (likely classification format)")
        for folder in class_folders[:10]:  # Show first 10
            print(f"  - {folder.name}")
        structures['classes'] = [f.name for f in class_folders]
    
    return structures

def convert_classification_to_yolo(dataset_path, output_path):
    """
    Convert classification dataset (images in class folders) to YOLO format.
    Handles nested folder structures.
    This assumes each image shows a hand making a sign - we'll create bounding boxes
    that cover most of the image (you may need to adjust this).
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Create YOLO directory structure
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # First, try to find class folders - they might be nested
    print("\nSearching for class folders and images...")
    
    # Strategy 1: Check if there's a single folder that contains class folders
    top_level_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    # If we found a single container folder (like "asl_dataset"), look inside it
    if len(top_level_folders) == 1 and top_level_folders[0].is_dir():
        print(f"Found container folder: {top_level_folders[0].name}")
        dataset_path = top_level_folders[0]  # Look inside it
        top_level_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    # Now get class folders (these should be the letter folders like A, B, C, etc.)
    class_folders = sorted([d for d in top_level_folders if d.is_dir()])
    
    if not class_folders:
        print("ERROR: No class folders found!")
        print(f"Looking in: {dataset_path}")
        print(f"Found folders: {[f.name for f in top_level_folders]}")
        print("\nTrying to find images recursively...")
        
        # Try to find all images recursively
        all_image_files = list(dataset_path.rglob('*.jpg')) + list(dataset_path.rglob('*.png')) + list(dataset_path.rglob('*.jpeg'))
        print(f"Found {len(all_image_files)} images total")
        if all_image_files:
            print("Sample image paths:")
            for img in all_image_files[:5]:
                print(f"  {img}")
            print("\nPlease check the dataset structure. Images found but no class folders detected.")
        else:
            print("No images found at all. Please check the dataset path.")
        return None
    
    class_names = [f.name for f in class_folders]
    print(f"Found {len(class_names)} class folders: {class_names[:10]}...")  # Show first 10
    
    # Create class mapping
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    # Process each class
    all_images = []
    for class_folder in class_folders:
        class_name = class_folder.name
        class_id = class_to_id[class_name]
        
        # Look for images in the class folder (including subdirectories)
        images = (list(class_folder.glob('*.jpg')) + 
                 list(class_folder.glob('*.png')) + 
                 list(class_folder.glob('*.jpeg')) +
                 list(class_folder.rglob('*.jpg')) + 
                 list(class_folder.rglob('*.png')) + 
                 list(class_folder.rglob('*.jpeg')))
        
        # Remove duplicates
        images = list(set(images))
        
        print(f"  {class_name}: {len(images)} images")
        
        for img_path in images:
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Create bounding box covering most of the image (center 80%)
            # YOLO format: normalized (x_center, y_center, width, height)
            x_center = 0.5
            y_center = 0.5
            box_width = 0.8
            box_height = 0.8
            
            all_images.append({
                'path': img_path,
                'class_id': class_id,
                'class_name': class_name,
                'bbox': (x_center, y_center, box_width, box_height)
            })
    
    if len(all_images) == 0:
        print("\nERROR: No images found in any class folders!")
        print("Please check the dataset structure.")
        return None
    
    print(f"\nTotal images found: {len(all_images)}")
    
    # Split into train/val
    if len(all_images) < 2:
        print("ERROR: Not enough images to split (need at least 2 images)")
        return None
    
    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)
    
    # Copy images and create label files
    for split, images in [('train', train_images), ('val', val_images)]:
        for img_data in images:
            # Copy image
            img_name = img_data['path'].name
            dest_img = output_path / 'images' / split / img_name
            shutil.copy(img_data['path'], dest_img)
            
            # Create label file
            label_name = img_data['path'].stem + '.txt'
            dest_label = output_path / 'labels' / split / label_name
            
            with open(dest_label, 'w') as f:
                x_center, y_center, box_width, box_height = img_data['bbox']
                f.write(f"{img_data['class_id']} {x_center} {y_center} {box_width} {box_height}\n")
    
    # Create data.yaml file
    yaml_content = f"""path: {output_path.absolute()}
train: images/train
val: images/val

names:
"""
    for idx, class_name in enumerate(class_names):
        yaml_content += f"  {idx}: {class_name}\n"
    
    with open(output_path / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"\nDataset converted! Found {len(class_names)} classes:")
    for name in class_names:
        print(f"  - {name}")
    print(f"\nTrain images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    print(f"\nYAML file created at: {output_path / 'data.yaml'}")
    
    return output_path / 'data.yaml'

# Run this to explore and convert
if __name__ == "__main__":
    import kagglehub
    
    # Download dataset
    path = kagglehub.dataset_download("ayuraj/asl-dataset")
    print(f"Dataset downloaded to: {path}")
    
    # Explore structure
    structures = explore_dataset(path)
    
    # If it's classification format, convert it
    if structures['classes']:
        print("\nConverting to YOLO format...")
        yaml_path = convert_classification_to_yolo(path, "asl_yolo_dataset")
        print(f"\n✅ Ready to train! Use this YAML: {yaml_path}")
    else:
        print("\nDataset might already be in YOLO format or different structure.")
        print("Please check the dataset structure manually.")