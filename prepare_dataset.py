import os
import pandas as pd
import faiss
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1

# **Prepare Dataset**
dataset_path = "Dataset"

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"Dataset directory '{dataset_path}' not found!")
    print("Please ensure the Dataset folder is in the same directory as this script.")
    exit(1)

print("Files in dataset:", os.listdir(dataset_path))

image_paths = []
labels = []

for filename in os.listdir(dataset_path):
    if filename.endswith((".jpg", ".JPG", ".png", ".jpeg")):
        image_paths.append(os.path.join(dataset_path, filename))
        file_name = filename.split(".")[0]
        label = file_name[7:]  # Remove "Avatar_" prefix
        labels.append(label)

print(f"Found {len(image_paths)} images")
df = pd.DataFrame({"image_path": image_paths, "label": labels})

# Store Image like vector

## Vectorize Image (Method 1: Raw pixels)
IMAGE_SIZE = 300
VECTOR_DIM = 300 * 300 * 3  # For RGB images (300x300x3)

index = faiss.IndexFlatL2(VECTOR_DIM)
label_map = []


def image_to_vector(image_path):
    """Convert image to normalized vector"""
    img = Image.open(image_path).resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img)

    # Handle grayscale images (convert to RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Normalize pixel values to [0, 1]
    vector = img_array.astype("float32") / 255.0
    return vector.flatten()


print("Building index with raw pixel features...")
for idx, row in df.iterrows():
    image_path = row["image_path"]
    label = row["label"]

    try:
        vector = image_to_vector(image_path)
        # Add to Faiss index
        index.add(np.array([vector]))
        label_map.append(label)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Save the index and label map for later use
faiss.write_index(index, "employee_images.index")
np.save("label_map.npy", np.array(label_map))


# Search functions for raw pixel method
def search_similar_images_raw(query_image_path, k=5):
    """Search for similar employee images using raw pixels"""
    # Load index and labels
    index = faiss.read_index("employee_images.index")
    label_map = np.load("label_map.npy")

    # Convert query image to vector
    query_vector = image_to_vector(query_image_path)

    # Search in Faiss
    distances, indices = index.search(np.array([query_vector]), k)

    # Get results
    results = []
    for i in range(len(indices[0])):
        employee_name = label_map[indices[0][i]]
        distance = distances[0][i]
        results.append((employee_name, distance))

    return results


def display_query_and_top_matches_raw(query_image_path):
    if not os.path.exists(query_image_path):
        print(f"Query image not found: {query_image_path}")
        return

    query_img = Image.open(query_image_path)
    query_img = query_img.resize((300, 300))

    plt.figure(figsize=(5, 5))
    plt.imshow(query_img)
    plt.title("Query Image (Raw Pixel Method)")
    plt.axis("off")
    plt.show()

    matches = search_similar_images_raw(query_image_path)

    # Display the top 5 matching employee images with distances
    top_matches = []
    for name, distance in matches:
        # Find the image path for this employee in df
        matching_rows = df[df["label"] == name]
        if not matching_rows.empty:
            img_path = matching_rows["image_path"].values[0]
            top_matches.append((name, distance, img_path))

    # Create plot
    plt.figure(figsize=(15, 5))
    for i, (name, distance, img_path) in enumerate(top_matches):
        img = Image.open(img_path)
        img = img.resize((300, 300))

        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(f"{name}\nDist: {distance:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Store feature of Image (Method 2: Deep learning features)

## Vectorize Image using FaceNet
print("Loading FaceNet model...")
try:
    face_recognition_model = InceptionResnetV1(pretrained="vggface2").eval()
except Exception as e:
    print(f"Error loading FaceNet model: {e}")
    print("Make sure facenet-pytorch is installed: pip install facenet-pytorch")
    exit(1)

transform = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def extract_feature(image_path, model):
    """Extract features from an image using a given model."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy()


# Build FaceNet index
VECTOR_DIM = 512
index_facenet = faiss.IndexFlatIP(VECTOR_DIM)
label_map_facenet = []

print("Building index with FaceNet features...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        features = extract_feature(row["image_path"], face_recognition_model)
        index_facenet.add(np.array([features]))
        label_map_facenet.append(row["label"])
    except Exception as e:
        print(f"Error processing {row['image_path']}: {e}")

# Save index and labels
faiss.write_index(index_facenet, "facenet_features.index")
np.save("facenet_label_map.npy", np.array(label_map_facenet))


# Search functions for FaceNet method
def image_to_feature(image_path, model):
    """Convert image to face embedding using a pre-trained model"""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.squeeze().numpy()


def search_similar_images_facenet(query_image_path, k=5):
    """Search for similar employee images using FaceNet features"""
    # Load index and labels
    index = faiss.read_index("facenet_features.index")
    label_map = np.load("facenet_label_map.npy")

    # Convert query image to vector
    query_vector = image_to_feature(query_image_path, face_recognition_model)

    # Search in Faiss
    similarities, indices = index.search(np.array([query_vector]), k)

    # Get results
    results = []
    for i in range(len(indices[0])):
        employee_name = label_map[indices[0][i]]
        similarity = similarities[0][i]
        results.append((employee_name, similarity))

    return results


def display_query_and_top_matches_facenet(query_image_path):
    if not os.path.exists(query_image_path):
        print(f"Query image not found: {query_image_path}")
        return

    # Display query image
    query_img = Image.open(query_image_path)
    query_img = query_img.resize((300, 300))
    plt.figure(figsize=(5, 5))
    plt.imshow(query_img)
    plt.title("Query Image (FaceNet Method)")
    plt.axis("off")
    plt.show()

    # Get matches
    matches = search_similar_images_facenet(query_image_path)

    # Display top matches
    plt.figure(figsize=(15, 5))
    for i, (name, similarity) in enumerate(matches):
        # Find the image path for this employee
        matching_rows = df[df["label"] == name]
        if not matching_rows.empty:
            img_path = matching_rows["image_path"].values[0]
            img = Image.open(img_path)
            img = img.resize((300, 300))

            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.title(f"{name}\nSimilarity: {similarity:.2f}")
            plt.axis("off")

    plt.tight_layout()
    plt.show()


# # Test with actual images from the dataset
# if len(df) > 0:
#     # Use the first image in the dataset as a test query
#     test_image_path = df.iloc[0]["image_path"]
#     print(f"\nTesting with image: {test_image_path}")
#
#     print("\n=== Raw Pixel Method ===")
#     display_query_and_top_matches_raw(test_image_path)
#
#     print("\n=== FaceNet Method ===")
#     display_query_and_top_matches_facenet(test_image_path)
#
#     # If you want to test with a specific employee
#     # Replace "Thuan_Duong" with an actual employee name from your dataset
#     target_employee = "Thuan_Duong"  # Change this to match your data
#     target_images = df[df["label"] == target_employee]
#
#     if not target_images.empty:
#         target_path = target_images.iloc[0]["image_path"]
#         print(f"\n=== Testing with {target_employee} ===")
#         print("Raw Pixel Method:")
#         display_query_and_top_matches_raw(target_path)
#         print("FaceNet Method:")
#         display_query_and_top_matches_facenet(target_path)
#     else:
#         print(f"Employee '{target_employee}' not found in dataset.")
#         print("Available employees:", df["label"].unique()[:10])  # Show first 10
# else:
#     print("No images found in dataset!")
