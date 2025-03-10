import cv2
import numpy as np
from ultralytics import YOLO

# ✅ Load YOLOv8 model (segmentation version)
model = YOLO("yolov8n-seg.pt")

# ✅ Load RealSense image & depth data
image_path = "/home/queen/test_image.jpg"
depth_image_path = "/home/queen/test_depth.npy"  # Load depth data
image = cv2.imread(image_path)
depth_image = np.load(depth_image_path)  # Depth stored as numpy array

if image is None or depth_image is None:
    print("Error: Could not load image or depth data.")
    exit()

# ✅ Run YOLOv8 segmentation on the image
results = model(image)

# ✅ Process bounding boxes & segmentation
for result in results:
    img = result.plot()  # Draw segmentation masks & bounding boxes

    for box, mask in zip(result.boxes.xyxy, result.masks.xy):
        x_min, y_min, x_max, y_max = map(int, box)

        # ✅ Bounding Box Depth (from center)
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        bounding_box_depth = depth_image[y_center, x_center] * 0.001  # Convert mm to meters

        # ✅ Segmentation Depth (Mean of Masked Pixels)
        mask = np.array(mask, dtype=np.uint8)  # Convert mask to proper format
        mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]))  # Resize to match depth image
        mask_indices = np.where(mask > 0)  # Get all non-zero mask pixels

        if len(mask_indices[0]) > 0:
            segmentation_depth = np.mean(depth_image[mask_indices]) * 0.001  # Convert mm to meters
        else:
            segmentation_depth = bounding_box_depth  # Fallback

        # ✅ Display depth values on the image
        cv2.putText(img, f"BB Depth: {bounding_box_depth:.2f}m", (x_min, y_min - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Seg Depth: {segmentation_depth:.2f}m", (x_min, y_min - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

# ✅ Show results
cv2.imshow("YOLOv8 Segmentation - Depth Measurement", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
