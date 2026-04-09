import cv2
import urllib.request
import numpy as np

from face.face_detector import FaceDetector
from face.face_encoder import FaceEncoder
from database.db import Database

detector = FaceDetector()
encoder = FaceEncoder()
db = Database()

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

print("👉 Add from Cloudinary URL")

while True:
    url = input("\nPaste URL (or 'q' to quit): ")
    if url == 'q':
        break

    img = url_to_image(url)
    face = detector.detect(img)

    if face is None:
        print("❌ No face found")
        continue

    emb = encoder.encode(face)

    name = input("Name: ")
    role = input("Role (student/teacher): ")

    if role == "student":
        roll = input("Roll: ")
        cls = input("Class: ")

        db.insert_student(name, roll, cls, url, emb)

    else:
        dept = input("Department: ")
        db.insert_teacher(name, dept, url, emb)

    print("✅ Saved!")