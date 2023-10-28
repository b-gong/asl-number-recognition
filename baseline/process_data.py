from PIL import Image
import numpy as np
import os

def load_and_preprocess_images(path, size=(64, 64)):
    """Load and preprocess images from a given path."""
    images = []
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize(size)  # Resize the image
        images.append(np.array(img))
    return images

def calculate_average_image(images):
    """Calculate the average image from a list of images."""
    return np.mean(images, axis=0)

# Load and preprocess train images
avg_images = {}
for number in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
    images = load_and_preprocess_images(os.path.join('../dataset', 'train', number))
    avg_images[number] = calculate_average_image(images)

def predict_number(test_img, avg_images):
    """Predict the number for a test image using the average images."""
    errors = {}
    for number, avg_img in avg_images.items():
        error = np.linalg.norm(test_img - avg_img)  # Calculate the Euclidean norm as error
        errors[number] = error
    return min(errors, key=errors.get)  # Return the number with the minimum error

# Test
test_images = {}
num_correct = 0
num_wrong = 0
for number in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
    images = load_and_preprocess_images(os.path.join('../dataset', 'test', number))
    for test_image in images:
        predicted_number = predict_number(test_image, avg_images)
        if (number == predicted_number):
            num_correct += 1
        else:
            num_wrong += 1
        print(f"The predicted number is: {predicted_number}")

num_total_images = num_correct + num_wrong
print(f"Accuracy on test set is {num_correct / num_total_images}")