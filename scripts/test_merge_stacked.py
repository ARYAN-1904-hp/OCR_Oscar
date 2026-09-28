import sys
sys.path.insert(0, '.')
from PIL import Image
import numpy as np
from ocr.pipeline import _paddle_words_on_image

img_path = r'c:\Users\ARYAN NAIR\Downloads\OCR_Enhancement_Task\test_dataset\samples\A-01_npi-vs-phone\provider_panel_1.png'
img = Image.open(img_path).convert('RGB')
arr = np.array(img)
words = _paddle_words_on_image(arr)

def merge_stacked(words):
    if not words:
        return []
    heights = sorted(w['bbox'][3] - w['bbox'][1] for w in words)
    med_h = heights[len(heights)//2] if heights else 12.0
    used = set()
    merged = []
    ws = sorted(words, key=lambda w: (w['bbox'][1], w['bbox'][0]))
    for i, w1 in enumerate(ws):
        if i in used:
            continue
        found_j = None
        for j in range(i + 1, len(ws)):
            if j in used:
                continue
            w2 = ws[j]
            v_gap = w2['bbox'][1] - w1['bbox'][3]
            # w2 directly below w1
            if -4 <= v_gap <= med_h * 0.8:
                w1_w = w1['bbox'][2] - w1['bbox'][0]
                w2_w = w2['bbox'][2] - w2['bbox'][0]
                x_overlap = min(w1['bbox'][2], w2['bbox'][2]) - max(w1['bbox'][0], w2['bbox'][0])
                min_w = min(w1_w, w2_w)
                left_align = abs(w1['bbox'][0] - w2['bbox'][0]) <= max(12, min_w * 0.4)
                if x_overlap > 0.3 * min_w or left_align:
                    comb_h = w2['bbox'][3] - w1['bbox'][1]
                    if comb_h <= med_h * 2.8:
                        found_j = j
                        break
        if found_j is not None:
            w2 = ws[found_j]
            used.add(found_j)
            merged.append({
                'text': f"{w1['text']} {w2['text']}",
                'conf': (w1.get('conf', 1.0) + w2.get('conf', 1.0)) / 2.0,
                'bbox': [min(w1['bbox'][0], w2['bbox'][0]), w1['bbox'][1], max(w1['bbox'][2], w2['bbox'][2]), w2['bbox'][3]]
            })
        else:
            merged.append(w1)
    return merged

m_words = merge_stacked(words)
print(f"Original words: {len(words)}, After merge: {len(m_words)}")
for wd in m_words:
    print(f"{wd['text']:35s} bbox={wd['bbox']}")
