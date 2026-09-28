import cv2
from pprint import pprint

from ocr.ppstructure_engine import PPStructureEngine

# Load image
image = cv2.imread("test_images/sample.png")

if image is None:
    raise FileNotFoundError("Could not load test_images/sample.png")

# Initialize engine
engine = PPStructureEngine()

print(f"PPStructure Available: {engine.is_available()}")

# Run analysis
result = engine.analyse(image)

print("\n========== RESULT ==========\n")

pprint(result)