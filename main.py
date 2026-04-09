import cv2

from face.face_detector import FaceDetector
from face.face_encoder import FaceEncoder
from face.face_matcher import FaceMatcher
from database.db import Database
from config import VIDEO_SOURCE

# -------------------------
detector = FaceDetector()
encoder = FaceEncoder()
db = Database()
matcher = FaceMatcher(db)

cap = cv2.VideoCapture(VIDEO_SOURCE)

FRAME_SKIP = 8
frame_count = 0

# 🔥 memory for stability
last_identity = None
last_embedding = None

print("\n🎥 Processing video...\n")

# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP != 0:
        continue

    faces = detector.detect(frame)

    if not faces:
        continue

    print(f"\nFrame {frame_count}")

    for face in faces:
        try:
            emb = encoder.encode(face)
            person = matcher.match(emb)

            # -------------------------
            # 🔥 FORCE STABILITY
            # -------------------------
            if person:
                last_identity = person
                last_embedding = emb

            elif last_embedding is not None:
                # fallback match using previous embedding
                dist = ((emb - last_embedding) ** 2).sum() ** 0.5

                if dist < 1.0:   # relaxed continuity threshold
                    person = last_identity

            # -------------------------
            # PRINT ONLY IF MATCHED
            # -------------------------
            if person:
                if person["role"] == "student":
                    print(f"🎓 {person['name']} | Roll:{person['roll']} | Class:{person['class']}")
                else:
                    print(f"👩‍🏫 {person['name']} | Dept:{person['dept']}")

        except:
            continue

cap.release()
print("\n✅ Done")