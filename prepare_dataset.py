# Import các thư viện cần thiết
import os  # Thao tác với hệ thống file và thư mục
import pandas as pd  # Xử lý dữ liệu dạng bảng (DataFrame)
import faiss  # Thư viện tìm kiếm vector similarity nhanh của Facebook
import numpy as np  # Thao tác với mảng số học
from PIL import Image  # Xử lý ảnh (mở, resize, chuyển đổi format)
import matplotlib.pyplot as plt  # Vẽ biểu đồ và hiển thị ảnh
import torch  # Framework deep learning PyTorch
import torchvision.models as models  # Các model deep learning có sẵn
import torchvision.transforms as transforms  # Biến đổi ảnh cho deep learning
from tqdm import tqdm  # Thanh tiến trình (progress bar)
from facenet_pytorch import InceptionResnetV1  # Model FaceNet để nhận diện khuôn mặt

# **Chuẩn bị Dataset**
dataset_path = "Dataset"  # Đường dẫn đến thư mục chứa ảnh

# Kiểm tra xem thư mục dataset có tồn tại không
if not os.path.exists(dataset_path):
    print(
        f"Dataset directory '{dataset_path}' not found!"
    )  # Thông báo lỗi nếu không tìm thấy
    print("Please ensure the Dataset folder is in the same directory as this script.")
    exit(1)  # Thoát chương trình với mã lỗi 1

print("Files in dataset:", os.listdir(dataset_path))  # In danh sách file trong thư mục

# Khởi tạo danh sách để lưu đường dẫn ảnh và nhãn
image_paths = []  # Danh sách đường dẫn đến các file ảnh
labels = []  # Danh sách tên nhân viên (nhãn)

# Duyệt qua tất cả file trong thư mục dataset
for filename in os.listdir(dataset_path):
    # Chỉ xử lý các file có định dạng ảnh
    if filename.endswith((".jpg", ".JPG", ".png", ".jpeg")):
        image_paths.append(
            os.path.join(dataset_path, filename)
        )  # Thêm đường dẫn đầy đủ vào danh sách
        file_name = filename.split(".")[0]  # Lấy tên file không có phần mở rộng
        label = file_name[7:]  # Bỏ tiền tố "Avatar_" để lấy tên nhân viên
        labels.append(label)  # Thêm tên nhân viên vào danh sách nhãn

print(f"Found {len(image_paths)} images")  # In số lượng ảnh tìm thấy
df = pd.DataFrame(
    {"image_path": image_paths, "label": labels}
)  # Tạo DataFrame chứa đường dẫn và nhãn

# Lưu trữ ảnh dưới dạng vector

## Phương pháp 1: Vectorize ảnh bằng pixel thô
IMAGE_SIZE = 300  # Kích thước ảnh sau khi resize (300x300)
VECTOR_DIM = 300 * 300 * 3  # Chiều của vector (300x300x3 cho ảnh RGB)

index = faiss.IndexFlatL2(
    VECTOR_DIM
)  # Tạo FAISS index sử dụng khoảng cách L2 (Euclidean)
label_map = []  # Danh sách ánh xạ index với tên nhân viên


def image_to_vector(image_path):
    """Chuyển đổi ảnh thành vector đã được chuẩn hóa"""
    img = Image.open(image_path).resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )  # Mở ảnh và resize về 300x300
    img_array = np.array(img)  # Chuyển ảnh thành mảng numpy

    # Xử lý ảnh xám (chuyển thành RGB)
    if len(img_array.shape) == 2:  # Nếu ảnh chỉ có 2 chiều (grayscale)
        img_array = np.stack((img_array,) * 3, axis=-1)  # Nhân đôi thành 3 kênh RGB

    # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    vector = img_array.astype("float32") / 255.0  # Chia cho 255 để chuẩn hóa
    return vector.flatten()  # Chuyển mảng 3D thành vector 1D


print("Building index with raw pixel features...")  # Thông báo đang xây dựng index
# Duyệt qua từng dòng trong DataFrame
for idx, row in df.iterrows():
    image_path = row["image_path"]  # Lấy đường dẫn ảnh
    label = row["label"]  # Lấy tên nhân viên

    try:
        vector = image_to_vector(image_path)  # Chuyển ảnh thành vector
        # Thêm vector vào FAISS index
        index.add(np.array([vector]))  # FAISS yêu cầu mảng 2D nên wrap trong []
        label_map.append(label)  # Thêm tên nhân viên vào danh sách ánh xạ
    except Exception as e:
        print(f"Error processing {image_path}: {e}")  # In lỗi nếu xử lý ảnh thất bại

# Lưu index và label map để sử dụng sau
faiss.write_index(index, "employee_images.index")  # Lưu FAISS index ra file
np.save("label_map.npy", np.array(label_map))  # Lưu danh sách ánh xạ ra file


# Hàm tìm kiếm cho phương pháp pixel thô
def search_similar_images_raw(query_image_path, k=5):
    """Tìm kiếm ảnh nhân viên tương tự sử dụng pixel thô"""
    # Load index và labels đã lưu
    index = faiss.read_index("employee_images.index")  # Đọc FAISS index từ file
    label_map = np.load("label_map.npy")  # Đọc danh sách ánh xạ từ file

    # Chuyển ảnh query thành vector
    query_vector = image_to_vector(query_image_path)

    # Tìm kiếm trong FAISS
    distances, indices = index.search(np.array([query_vector]), k)  # Tìm k ảnh gần nhất

    # Lấy kết quả
    results = []
    for i in range(len(indices[0])):  # Duyệt qua các kết quả tìm được
        employee_name = label_map[indices[0][i]]  # Lấy tên nhân viên từ index
        distance = distances[0][i]  # Lấy khoảng cách
        results.append((employee_name, distance))  # Thêm vào danh sách kết quả

    return results


def display_query_and_top_matches_raw(query_image_path):
    """Hiển thị ảnh query và top matches cho phương pháp pixel thô"""
    if not os.path.exists(query_image_path):  # Kiểm tra file tồn tại
        print(f"Query image not found: {query_image_path}")
        return

    # Hiển thị ảnh query
    query_img = Image.open(query_image_path)  # Mở ảnh query
    query_img = query_img.resize((300, 300))  # Resize về 300x300

    plt.figure(figsize=(5, 5))  # Tạo figure với kích thước 5x5
    plt.imshow(query_img)  # Hiển thị ảnh
    plt.title("Query Image (Raw Pixel Method)")  # Đặt tiêu đề
    plt.axis("off")  # Tắt trục tọa độ
    plt.show()  # Hiển thị plot

    matches = search_similar_images_raw(query_image_path)  # Tìm ảnh tương tự

    # Hiển thị top 5 ảnh nhân viên khớp nhất với khoảng cách
    top_matches = []
    for name, distance in matches:  # Duyệt qua các kết quả
        # Tìm đường dẫn ảnh cho nhân viên này trong df
        matching_rows = df[df["label"] == name]  # Lọc dòng có tên nhân viên khớp
        if not matching_rows.empty:  # Nếu tìm thấy
            img_path = matching_rows["image_path"].values[
                0
            ]  # Lấy đường dẫn ảnh đầu tiên
            top_matches.append(
                (name, distance, img_path)
            )  # Thêm vào danh sách hiển thị

    # Tạo plot hiển thị kết quả
    plt.figure(figsize=(15, 5))  # Tạo figure rộng để hiển thị 5 ảnh
    for i, (name, distance, img_path) in enumerate(
        top_matches
    ):  # Duyệt qua top matches
        img = Image.open(img_path)  # Mở ảnh
        img = img.resize((300, 300))  # Resize

        plt.subplot(1, 5, i + 1)  # Tạo subplot (1 hàng, 5 cột, vị trí i+1)
        plt.imshow(img)  # Hiển thị ảnh
        plt.title(f"{name}\nDist: {distance:.2f}")  # Tiêu đề với tên và khoảng cách
        plt.axis("off")  # Tắt trục

    plt.tight_layout()  # Tự động điều chỉnh layout
    plt.show()  # Hiển thị


# Lưu trữ đặc trưng của ảnh (Phương pháp 2: Deep learning features)

## Vectorize ảnh sử dụng FaceNet
print("Loading FaceNet model...")  # Thông báo đang load model
try:
    # Tạo model FaceNet đã được train trên VGGFace2 dataset
    face_recognition_model = InceptionResnetV1(
        pretrained="vggface2"
    ).eval()  # .eval() để chuyển về chế độ evaluation
except Exception as e:
    print(f"Error loading FaceNet model: {e}")
    print("Make sure facenet-pytorch is installed: pip install facenet-pytorch")
    exit(1)

# Định nghĩa các phép biến đổi ảnh cho FaceNet
transform = transforms.Compose(
    [
        transforms.Resize((300, 300)),  # Resize ảnh về 300x300
        transforms.ToTensor(),  # Chuyển ảnh thành tensor PyTorch
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ),  # Chuẩn hóa về [-1, 1]
    ]
)


def extract_feature(image_path, model):
    """Trích xuất đặc trưng từ ảnh sử dụng model đã cho."""
    img = Image.open(image_path).convert("RGB")  # Mở ảnh và chuyển về RGB
    img_tensor = transform(img).unsqueeze(
        0
    )  # Áp dụng transform và thêm batch dimension
    with torch.no_grad():  # Tắt gradient computation để tiết kiệm memory
        features = model(img_tensor)  # Chạy model để lấy features
    return features.squeeze().numpy()  # Bỏ batch dimension và chuyển về numpy


# Xây dựng FaceNet index
VECTOR_DIM = 512  # FaceNet tạo ra vector 512 chiều
index_facenet = faiss.IndexFlatIP(
    VECTOR_DIM
)  # Sử dụng Inner Product (cosine similarity)
label_map_facenet = []  # Danh sách ánh xạ cho FaceNet

print("Building index with FaceNet features...")
# Duyệt qua DataFrame với thanh tiến trình
for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        # Trích xuất features từ ảnh
        features = extract_feature(row["image_path"], face_recognition_model)
        index_facenet.add(np.array([features]))  # Thêm vào FAISS index
        label_map_facenet.append(row["label"])  # Thêm label vào ánh xạ
    except Exception as e:
        print(f"Error processing {row['image_path']}: {e}")

# Lưu index và labels
faiss.write_index(index_facenet, "facenet_features.index")  # Lưu FAISS index
np.save("facenet_label_map.npy", np.array(label_map_facenet))  # Lưu ánh xạ labels


# Hàm tìm kiếm cho phương pháp FaceNet
def image_to_feature(image_path, model):
    """Chuyển ảnh thành face embedding sử dụng pre-trained model"""
    img = Image.open(image_path).convert("RGB")  # Mở và chuyển về RGB
    img_tensor = transform(img).unsqueeze(0)  # Transform và thêm batch dimension
    with torch.no_grad():  # Tắt gradient
        embedding = model(img_tensor)  # Lấy embedding
    return embedding.squeeze().numpy()  # Trả về numpy array


def search_similar_images_facenet(query_image_path, k=5):
    """Tìm kiếm ảnh nhân viên tương tự sử dụng FaceNet features"""
    # Load index và labels
    index = faiss.read_index("facenet_features.index")  # Đọc FAISS index
    label_map = np.load("facenet_label_map.npy")  # Đọc ánh xạ labels

    # Chuyển ảnh query thành vector
    query_vector = image_to_feature(query_image_path, face_recognition_model)

    # Tìm kiếm trong FAISS
    similarities, indices = index.search(
        np.array([query_vector]), k
    )  # Tìm k ảnh tương tự nhất

    # Lấy kết quả
    results = []
    for i in range(len(indices[0])):  # Duyệt qua kết quả
        employee_name = label_map[indices[0][i]]  # Lấy tên nhân viên
        similarity = similarities[0][i]  # Lấy độ tương tự (cao hơn = tương tự hơn)
        results.append((employee_name, similarity))

    return results


def display_query_and_top_matches_facenet(query_image_path):
    """Hiển thị ảnh query và top matches cho phương pháp FaceNet"""
    if not os.path.exists(query_image_path):  # Kiểm tra file tồn tại
        print(f"Query image not found: {query_image_path}")
        return

    # Hiển thị ảnh query
    query_img = Image.open(query_image_path)
    query_img = query_img.resize((300, 300))
    plt.figure(figsize=(5, 5))
    plt.imshow(query_img)
    plt.title("Query Image (FaceNet Method)")  # Tiêu đề cho phương pháp FaceNet
    plt.axis("off")
    plt.show()

    # Lấy matches
    matches = search_similar_images_facenet(query_image_path)

    # Hiển thị top matches
    plt.figure(figsize=(15, 5))
    for i, (name, similarity) in enumerate(matches):  # Duyệt qua kết quả
        # Tìm đường dẫn ảnh cho nhân viên này
        matching_rows = df[df["label"] == name]
        if not matching_rows.empty:
            img_path = matching_rows["image_path"].values[0]
            img = Image.open(img_path)
            img = img.resize((300, 300))

            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.title(
                f"{name}\nSimilarity: {similarity:.2f}"
            )  # Hiển thị độ tương tự thay vì khoảng cách
            plt.axis("off")

    plt.tight_layout()
    plt.show()


# # Test với ảnh thực từ dataset
# if len(df) > 0:  # Nếu có ảnh trong dataset
#     # Sử dụng ảnh đầu tiên làm test query
#     test_image_path = df.iloc[0]["image_path"]
#     print(f"\nTesting with image: {test_image_path}")
#
#     print("\n=== Raw Pixel Method ===")  # Test phương pháp pixel thô
#     display_query_and_top_matches_raw(test_image_path)
#
#     print("\n=== FaceNet Method ===")  # Test phương pháp FaceNet
#     display_query_and_top_matches_facenet(test_image_path)
#
#     # Nếu muốn test với nhân viên cụ thể
#     # Thay "Thuan_Duong" bằng tên nhân viên thực tế trong dataset
#     target_employee = "Thuan_Duong"  # Thay đổi tên này cho phù hợp với dữ liệu
#     target_images = df[df["label"] == target_employee]  # Lọc ảnh của nhân viên đích
#
#     if not target_images.empty:  # Nếu tìm thấy nhân viên
#         target_path = target_images.iloc[0]["image_path"]  # Lấy ảnh đầu tiên
#         print(f"\n=== Testing with {target_employee} ===")
#         print("Raw Pixel Method:")
#         display_query_and_top_matches_raw(target_path)
#         print("FaceNet Method:")
#         display_query_and_top_matches_facenet(target_path)
#     else:
#         print(f"Employee '{target_employee}' not found in dataset.")
#         print("Available employees:", df["label"].unique()[:10])  # Hiển thị 10 nhân viên đầu
# else:
#     print("No images found in dataset!")  # Không tìm thấy ảnh nào
