from pprint import pprint
import cv2

from ocr.ppstructure_engine import PPStructureEngine
from ocr.layout_adapter import LayoutAdapter

img = cv2.imread("test_images/new_sample1.jpg")

engine = PPStructureEngine()

result = engine.analyse(img)

adapter = LayoutAdapter()

layout = adapter.from_ppstructure(result)

print("=" * 50)

print("Blocks :", len(layout.blocks))
print("Tables :", len(layout.tables))
print("Headers:", len(layout.headers))
print("Figures:", len(layout.figures))

print("=" * 50)

for block in layout.blocks:
    pprint(block)