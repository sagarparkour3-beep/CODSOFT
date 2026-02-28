import sys
print(f"Python ver: {sys.version}")
try:
    import mediapipe as mp
    print(f"MediaPipe imported. Version: {getattr(mp, '__version__', 'unknown')}")
    print(f"Has 'solutions': {'solutions' in dir(mp)}")
except ImportError:
    print("MediaPipe not installed.")

try:
    from mediapipe.python.solutions import face_detection
    print("Import from mediapipe.python.solutions.face_detection successful.")
except ImportError:
    print("Import from mediapipe.python.solutions.face_detection failed.")

try:
    import mediapipe.solutions.face_detection as fd
    print("Import mediapipe.solutions.face_detection successful.")
except Exception as e:
    print(f"Import mediapipe.solutions.face_detection failed: {e}")

try:
    print("Importing deepface...")
    from deepface import DeepFace
    print("Import deepface successful.")
except Exception as e:
    print(f"Import deepface failed: {e}")
