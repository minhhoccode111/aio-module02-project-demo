import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
import sqlite3
import json
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vector Database Interface (có thể thay thế bằng Pinecone, Weaviate, Chroma, etc.)
try:
    import chromadb
    from chromadb.config import Settings

    VECTOR_DB_AVAILABLE = True
except ImportError:
    logger.warning("ChromaDB không có sẵn. Sử dụng pip install chromadb")
    VECTOR_DB_AVAILABLE = False


@dataclass
class Student:
    """Data class đại diện cho một học viên"""

    student_id: str
    name: str
    class_id: str
    subject_ids: List[str]
    enrollment_date: datetime
    status: str = "active"  # active, graduated, suspended
    face_embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Chuyển đổi thành dictionary để lưu vào database"""
        data = asdict(self)
        data["enrollment_date"] = self.enrollment_date.isoformat()
        if self.face_embedding is not None:
            data["face_embedding"] = self.face_embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "Student":
        """Tạo Student object từ dictionary"""
        data["enrollment_date"] = datetime.fromisoformat(data["enrollment_date"])
        if data.get("face_embedding"):
            data["face_embedding"] = np.array(data["face_embedding"])
        return cls(**data)


@dataclass
class AttendanceRecord:
    """Data class cho bản ghi điểm danh"""

    record_id: str
    student_id: str
    class_id: str
    subject_id: str
    timestamp: datetime
    confidence_score: float
    status: str = "present"  # present, absent, late


class FaceEmbeddingExtractor:
    """Class trích xuất face embedding sử dụng FaceNet"""

    def __init__(self):
        """Khởi tạo FaceNet model"""
        logger.info("Đang tải FaceNet model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # Transform pipeline cho ảnh đầu vào
        self.transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),  # FaceNet input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        logger.info(f"FaceNet model loaded on {self.device}")

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """Trích xuất face embedding từ ảnh"""
        try:
            # Mở và xử lý ảnh
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Trích xuất embedding
            with torch.no_grad():
                embedding = self.model(img_tensor)

            # Normalize embedding để tối ưu cosine similarity
            embedding = embedding.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Lỗi khi trích xuất embedding từ {image_path}: {e}")
            raise

    def extract_embedding_from_image(self, image: Image.Image) -> np.ndarray:
        """Trích xuất embedding từ PIL Image object"""
        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(img_tensor)

            embedding = embedding.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Lỗi khi trích xuất embedding từ image: {e}")
            raise


class DatabaseManager:
    """Quản lý database cho metadata của học viên"""

    def __init__(self, db_path: str = "attendance_system.db"):
        """Khởi tạo SQLite database"""
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Tạo các bảng cần thiết"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Bảng học viên
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    class_id TEXT NOT NULL,
                    subject_ids TEXT NOT NULL,  -- JSON array
                    enrollment_date TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Bảng điểm danh
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance_records (
                    record_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    class_id TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    status TEXT DEFAULT 'present',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students (student_id)
                )
            """)

            # Bảng lớp học
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classes (
                    class_id TEXT PRIMARY KEY,
                    class_name TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    instructor TEXT,
                    schedule TEXT,  -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Tạo index để tối ưu query
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_student_class ON students(class_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance_records(student_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance_records(timestamp)"
            )

            conn.commit()
            logger.info("Database initialized successfully")

    def add_student(self, student: Student) -> bool:
        """Thêm học viên mới"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO students (student_id, name, class_id, subject_ids, enrollment_date, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        student.student_id,
                        student.name,
                        student.class_id,
                        json.dumps(student.subject_ids),
                        student.enrollment_date.isoformat(),
                        student.status,
                    ),
                )
                conn.commit()
                logger.info(
                    f"Đã thêm học viên {student.name} (ID: {student.student_id})"
                )
                return True
        except sqlite3.IntegrityError:
            logger.error(f"Học viên với ID {student.student_id} đã tồn tại")
            return False
        except Exception as e:
            logger.error(f"Lỗi khi thêm học viên: {e}")
            return False

    def update_student_status(self, student_id: str, status: str) -> bool:
        """Cập nhật trạng thái học viên"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE students
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE student_id = ?
                """,
                    (status, student_id),
                )

                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(
                        f"Đã cập nhật trạng thái học viên {student_id} thành {status}"
                    )
                    return True
                else:
                    logger.warning(f"Không tìm thấy học viên với ID {student_id}")
                    return False
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật trạng thái: {e}")
            return False

    def get_active_students_by_class(self, class_id: str) -> List[Dict]:
        """Lấy danh sách học viên active trong một lớp"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM students
                    WHERE class_id = ? AND status = 'active'
                    ORDER BY name
                """,
                    (class_id,),
                )

                columns = [description[0] for description in cursor.description]
                students = []
                for row in cursor.fetchall():
                    student_dict = dict(zip(columns, row))
                    student_dict["subject_ids"] = json.loads(
                        student_dict["subject_ids"]
                    )
                    students.append(student_dict)

                return students
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách học viên: {e}")
            return []

    def record_attendance(self, record: AttendanceRecord) -> bool:
        """Ghi lại bản ghi điểm danh"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO attendance_records
                    (record_id, student_id, class_id, subject_id, timestamp, confidence_score, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record.record_id,
                        record.student_id,
                        record.class_id,
                        record.subject_id,
                        record.timestamp.isoformat(),
                        record.confidence_score,
                        record.status,
                    ),
                )
                conn.commit()
                logger.info(f"Đã ghi điểm danh cho học viên {record.student_id}")
                return True
        except Exception as e:
            logger.error(f"Lỗi khi ghi điểm danh: {e}")
            return False


class VectorDatabaseManager:
    """Quản lý vector database cho face embeddings"""

    def __init__(self, collection_name: str = "student_faces"):
        """Khởi tạo ChromaDB client"""
        if not VECTOR_DB_AVAILABLE:
            raise ImportError(
                "ChromaDB không có sẵn. Cài đặt với: pip install chromadb"
            )

        # Khởi tạo ChromaDB client
        self.client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db")
        )

        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Sử dụng cosine similarity
        )

        logger.info(f"Vector database initialized with collection: {collection_name}")

    def add_student_embedding(
        self, student_id: str, embedding: np.ndarray, metadata: Dict
    ) -> bool:
        """Thêm face embedding của học viên"""
        try:
            self.collection.add(
                embeddings=[embedding.tolist()], metadatas=[metadata], ids=[student_id]
            )
            logger.info(f"Đã thêm embedding cho học viên {student_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi thêm embedding: {e}")
            return False

    def search_similar_faces(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        class_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Tìm kiếm khuôn mặt tương tự"""
        try:
            # Tạo where clause để filter theo class nếu cần
            where_clause = None
            if class_filter:
                where_clause = {"class_id": class_filter}

            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_clause,
                include=["metadatas", "distances"],
            )

            # Format kết quả
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i, student_id in enumerate(results["ids"][0]):
                    formatted_results.append(
                        {
                            "student_id": student_id,
                            "similarity": 1
                            - results["distances"][0][
                                i
                            ],  # Convert distance to similarity
                            "metadata": results["metadatas"][0][i],
                        }
                    )

            return formatted_results

        except Exception as e:
            logger.error(f"Lỗi khi search: {e}")
            return []

    def remove_student(self, student_id: str) -> bool:
        """Xóa embedding của học viên"""
        try:
            self.collection.delete(ids=[student_id])
            logger.info(f"Đã xóa embedding của học viên {student_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa embedding: {e}")
            return False

    def update_student_metadata(self, student_id: str, new_metadata: Dict) -> bool:
        """Cập nhật metadata của học viên"""
        try:
            # ChromaDB không hỗ trợ update trực tiếp, cần delete và add lại
            # Lấy embedding hiện tại
            result = self.collection.get(ids=[student_id], include=["embeddings"])
            if result["ids"]:
                embedding = result["embeddings"][0]
                self.collection.delete(ids=[student_id])
                self.collection.add(
                    embeddings=[embedding], metadatas=[new_metadata], ids=[student_id]
                )
                logger.info(f"Đã cập nhật metadata cho học viên {student_id}")
                return True
            else:
                logger.warning(f"Không tìm thấy embedding cho học viên {student_id}")
                return False
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật metadata: {e}")
            return False


class AttendanceSystem:
    """Hệ thống điểm danh chính"""

    def __init__(self):
        """Khởi tạo hệ thống"""
        self.face_extractor = FaceEmbeddingExtractor()
        self.db_manager = DatabaseManager()

        if VECTOR_DB_AVAILABLE:
            self.vector_db = VectorDatabaseManager()
        else:
            logger.warning("Vector database không khả dụng, chỉ sử dụng SQLite")
            self.vector_db = None

        # Threshold để xác định match
        self.similarity_threshold = 0.6

        logger.info("Hệ thống điểm danh đã sẵn sàng")

    def enroll_student(
        self,
        student_id: str,
        name: str,
        class_id: str,
        subject_ids: List[str],
        image_path: str,
    ) -> bool:
        """Đăng ký học viên mới với ảnh khuôn mặt"""
        try:
            # Tạo object Student
            student = Student(
                student_id=student_id,
                name=name,
                class_id=class_id,
                subject_ids=subject_ids,
                enrollment_date=datetime.now(),
            )

            # Trích xuất face embedding
            embedding = self.face_extractor.extract_embedding(image_path)

            # Lưu vào SQLite
            if not self.db_manager.add_student(student):
                return False

            # Lưu embedding vào vector database
            if self.vector_db:
                metadata = {
                    "name": name,
                    "class_id": class_id,
                    "subject_ids": subject_ids,
                    "enrollment_date": student.enrollment_date.isoformat(),
                }
                if not self.vector_db.add_student_embedding(
                    student_id, embedding, metadata
                ):
                    # Rollback nếu thất bại
                    self.db_manager.update_student_status(student_id, "inactive")
                    return False

            logger.info(f"Đã đăng ký thành công học viên {name}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi đăng ký học viên: {e}")
            return False

    def graduate_student(self, student_id: str) -> bool:
        """Chuyển học viên sang trạng thái graduated"""
        try:
            # Cập nhật status trong SQLite
            if not self.db_manager.update_student_status(student_id, "graduated"):
                return False

            # Xóa khỏi vector database (không cần search nữa)
            if self.vector_db:
                self.vector_db.remove_student(student_id)

            logger.info(f"Đã chuyển học viên {student_id} sang trạng thái graduated")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi graduate học viên: {e}")
            return False

    def recognize_and_mark_attendance(
        self, image_path: str, class_id: str, subject_id: str
    ) -> Dict:
        """Nhận diện khuôn mặt và điểm danh"""
        try:
            # Trích xuất embedding từ ảnh đầu vào
            query_embedding = self.face_extractor.extract_embedding(image_path)

            if not self.vector_db:
                return {"error": "Vector database không khả dụng"}

            # Tìm kiếm trong vector database với filter theo class
            results = self.vector_db.search_similar_faces(
                query_embedding,
                n_results=3,  # Lấy top 3 để so sánh
                class_filter=class_id,
            )

            if not results:
                return {
                    "status": "no_match",
                    "message": "Không tìm thấy khuôn mặt phù hợp",
                }

            # Kiểm tra similarity threshold
            best_match = results[0]
            if best_match["similarity"] < self.similarity_threshold:
                return {
                    "status": "low_confidence",
                    "message": f"Độ tương tự thấp: {best_match['similarity']:.3f}",
                    "candidates": results,
                }

            # Ghi điểm danh
            record = AttendanceRecord(
                record_id=f"{best_match['student_id']}_{datetime.now().timestamp()}",
                student_id=best_match["student_id"],
                class_id=class_id,
                subject_id=subject_id,
                timestamp=datetime.now(),
                confidence_score=best_match["similarity"],
            )

            if self.db_manager.record_attendance(record):
                return {
                    "status": "success",
                    "student_id": best_match["student_id"],
                    "student_name": best_match["metadata"]["name"],
                    "confidence": best_match["similarity"],
                    "timestamp": record.timestamp.isoformat(),
                }
            else:
                return {"status": "error", "message": "Lỗi khi ghi điểm danh"}

        except Exception as e:
            logger.error(f"Lỗi khi nhận diện: {e}")
            return {"status": "error", "message": str(e)}

    def get_class_attendance_report(
        self, class_id: str, subject_id: str, date: datetime
    ) -> Dict:
        """Lấy báo cáo điểm danh của lớp trong ngày"""
        try:
            # Lấy danh sách học viên active trong lớp
            students = self.db_manager.get_active_students_by_class(class_id)

            # Lấy bản ghi điểm danh trong ngày
            start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)

            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT student_id, timestamp, confidence_score, status
                    FROM attendance_records
                    WHERE class_id = ? AND subject_id = ?
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """,
                    (
                        class_id,
                        subject_id,
                        start_date.isoformat(),
                        end_date.isoformat(),
                    ),
                )

                attendance_records = cursor.fetchall()

            # Tạo báo cáo
            report = {
                "class_id": class_id,
                "subject_id": subject_id,
                "date": date.date().isoformat(),
                "total_students": len(students),
                "present_count": len(attendance_records),
                "absent_count": len(students) - len(attendance_records),
                "students": [],
            }

            # Danh sách học viên có mặt
            present_students = {record[0] for record in attendance_records}

            for student in students:
                student_report = {
                    "student_id": student["student_id"],
                    "name": student["name"],
                    "status": "present"
                    if student["student_id"] in present_students
                    else "absent",
                }

                # Thêm thông tin chi tiết nếu có mặt
                if student["student_id"] in present_students:
                    for record in attendance_records:
                        if record[0] == student["student_id"]:
                            student_report.update(
                                {
                                    "check_in_time": record[1],
                                    "confidence_score": record[2],
                                }
                            )
                            break

                report["students"].append(student_report)

            return report

        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo: {e}")
            return {"error": str(e)}


# Example usage và demo
def demo_system():
    """Demo cách sử dụng hệ thống"""
    if not VECTOR_DB_AVAILABLE:
        print("Cần cài đặt ChromaDB để chạy demo: pip install chromadb")
        return

    # Khởi tạo hệ thống
    system = AttendanceSystem()

    print("=== DEMO HỆ THỐNG ĐIỂM DANH ===")

    # 1. Đăng ký học viên (giả lập)
    print("\n1. Đăng ký học viên mới...")
    """
    # Uncomment để test với ảnh thực
    success = system.enroll_student(
        student_id="SV001",
        name="Nguyen Van A",
        class_id="CLASS_CS101",
        subject_ids=["MATH101", "CS101"],
        image_path="path/to/student/photo.jpg"
    )
    print(f"Đăng ký học viên: {'Thành công' if success else 'Thất bại'}")
    """

    # 2. Điểm danh (giả lập)
    print("\n2. Thực hiện điểm danh...")
    """
    result = system.recognize_and_mark_attendance(
        image_path="path/to/query/photo.jpg",
        class_id="CLASS_CS101",
        subject_id="CS101"
    )
    print(f"Kết quả điểm danh: {result}")
    """

    # 3. Lấy báo cáo
    print("\n3. Tạo báo cáo điểm danh...")
    """
    report = system.get_class_attendance_report(
        class_id="CLASS_CS101",
        subject_id="CS101",
        date=datetime.now()
    )
    print(f"Báo cáo: {report}")
    """

    # 4. Graduate học viên
    print("\n4. Graduate học viên...")
    """
    success = system.graduate_student("SV001")
    print(f"Graduate học viên: {'Thành công' if success else 'Thất bại'}")
    """

    print("\nDemo hoàn thành! Uncomment các phần code để test với dữ liệu thực.")


if __name__ == "__main__":
    demo_system()
