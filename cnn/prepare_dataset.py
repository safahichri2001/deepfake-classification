import os
import random
import shutil

base = "."

train_real = os.path.join(base, "train", "real")
train_fake = os.path.join(base, "train", "fake")

val_real = os.path.join(base, "validation", "real")
val_fake = os.path.join(base, "validation", "fake")

test_real = os.path.join(base, "test", "real")
test_fake = os.path.join(base, "test", "fake")

os.makedirs(test_real, exist_ok=True)
os.makedirs(test_fake, exist_ok=True)

TRAIN_SIZE = 1500
VAL_SIZE = 300
TEST_SIZE = 300


def move_random_images(source, destination, number):
    images = os.listdir(source)
    selected = random.sample(images, min(number, len(images)))
    
    for img in selected:
        shutil.move(
            os.path.join(source, img),
            os.path.join(destination, img)
        )


def reduce_dataset(folder, max_images):
    images = os.listdir(folder)
    
    if len(images) > max_images:
        remove = random.sample(images, len(images) - max_images)
        
        for img in remove:
            os.remove(os.path.join(folder, img))


print("Creating test dataset...")

move_random_images(train_real, test_real, TEST_SIZE)
move_random_images(train_fake, test_fake, TEST_SIZE)

print("Reducing train dataset...")

reduce_dataset(train_real, TRAIN_SIZE)
reduce_dataset(train_fake, TRAIN_SIZE)

print("Reducing validation dataset...")

reduce_dataset(val_real, VAL_SIZE)
reduce_dataset(val_fake, VAL_SIZE)

print("Dataset ready 🚀")