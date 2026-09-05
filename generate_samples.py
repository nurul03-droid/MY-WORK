import os
import cv2
import numpy as np

def generate_samples(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Single rectangle
    img1 = np.full((500, 500), 255, dtype=np.uint8)
    cv2.rectangle(img1, (100, 100), (400, 400), 0, thickness=3)
    cv2.imwrite(os.path.join(output_dir, "single_rect.png"), img1)
    
    # 2. Two adjacent rectangles sharing an edge
    img2 = np.full((500, 500), 255, dtype=np.uint8)
    cv2.rectangle(img2, (100, 100), (250, 400), 0, thickness=3)
    cv2.rectangle(img2, (250, 100), (400, 400), 0, thickness=3)
    cv2.imwrite(os.path.join(output_dir, "adjacent_rects.png"), img2)
    
    # 3. L-shaped plot
    img3 = np.full((500, 500), 255, dtype=np.uint8)
    pts = np.array([[100, 100], [250, 100], [250, 250], [400, 250], [400, 400], [100, 400]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img3, [pts], isClosed=True, color=0, thickness=3)
    cv2.imwrite(os.path.join(output_dir, "l_shape.png"), img3)
    
    # 4. Broken/gapped boundary line
    img4 = np.full((500, 500), 255, dtype=np.uint8)
    cv2.line(img4, (100, 100), (400, 100), 0, thickness=3)
    cv2.line(img4, (400, 100), (400, 400), 0, thickness=3)
    cv2.line(img4, (400, 400), (100, 400), 0, thickness=3)
    # The left edge has a gap in the middle
    cv2.line(img4, (100, 400), (100, 280), 0, thickness=3)
    cv2.line(img4, (100, 220), (100, 100), 0, thickness=3)
    cv2.imwrite(os.path.join(output_dir, "broken_boundary.png"), img4)

if __name__ == "__main__":
    generate_samples("sample_data")
