import os
import random
import shutil

# Path to your dataset directory
source_path = '../assets'
dataset_path = '../dataset'

# Proportions for train, dev, and test
TRAIN_PROPORTION = 0.7
DEV_PROPORTION = 0.2
TEST_PROPORTION = 0.1

# Creating directories for train, dev, and test sets
train_dir = os.path.join(dataset_path, 'train')
dev_dir = os.path.join(dataset_path, 'dev')
test_dir = os.path.join(dataset_path, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(dev_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to split files into train, dev, and test sets
def split_files(source_folder, train_dest, dev_dest, test_dest):
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    random.shuffle(files)
    
    train_count = int(len(files) * TRAIN_PROPORTION)
    dev_count = int(len(files) * DEV_PROPORTION)
    
    for i, file in enumerate(files):
        source = os.path.join(source_folder, file)
        if i < train_count:
            shutil.copy(source, os.path.join(train_dest, file))
        elif i < train_count + dev_count:
            shutil.copy(source, os.path.join(dev_dest, file))
        else:
            shutil.copy(source, os.path.join(test_dest, file))

# Splitting images for each label
for folder in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'unknown']:
    source_folder = os.path.join(source_path, folder)
    
    train_dest = os.path.join(train_dir, folder)
    dev_dest = os.path.join(dev_dir, folder)
    test_dest = os.path.join(test_dir, folder)
    
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(dev_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)

    split_files(source_folder, train_dest, dev_dest, test_dest)

print("Splitting complete!")