import sys
sys.path.insert(0, '.')
from PIL import Image
import numpy as np
from ocr.pipeline import _paddle_words_on_image

def test_pipeline_on_image(img_path):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    page_h, page_w = arr.shape[:2]
    words = _paddle_words_on_image(arr)

    # 1. Stacked label merger
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

    # 2. Group into visual horizontal rows by y-overlap
    # Sort by y center
    def y_center(w):
        return (w['bbox'][1] + w['bbox'][3]) / 2.0

    ws_sorted = sorted(merged, key=lambda w: (y_center(w), w['bbox'][0]))
    thr = max(6.0, med_h * 0.75)
    rows = []
    current = []
    current_y = None

    for w in ws_sorted:
        yc = y_center(w)
        if current_y is None:
            current = [w]
            current_y = yc
            continue
        # Check overlap with current row's bounding box
        cur_y1 = min(x['bbox'][1] for x in current)
        cur_y2 = max(x['bbox'][3] for x in current)
        overlap = min(w['bbox'][3], cur_y2) - max(w['bbox'][1], cur_y1)
        w_h = w['bbox'][3] - w['bbox'][1]
        
        if abs(yc - current_y) <= thr or (overlap > 0.4 * w_h and abs(yc - current_y) <= med_h * 1.2):
            current.append(w)
            n = len(current)
            current_y = (current_y * (n - 1) + yc) / n
        else:
            rows.append(sorted(current, key=lambda x: x['bbox'][0]))
            current = [w]
            current_y = yc

    if current:
        rows.append(sorted(current, key=lambda x: x['bbox'][0]))

    print(f"\n===== RESULTS FOR: {img_path} =====")
    for r_idx, row in enumerate(rows):
        line_str = "  |  ".join(w['text'] for w in row) if any(w['text'].endswith(':') for w in row) else " ".join(w['text'] for w in row)
        # Or let's see how tokens in row form natural label-value pairs
        print(f"Row {r_idx:2d}: {line_str}")

for i in range(1, 6):
    test_pipeline_on_image(rf'c:\Users\ARYAN NAIR\Downloads\OCR_Enhancement_Task\test_dataset\samples\A-01_npi-vs-phone\provider_panel_{i}.png')
