import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (segmentation version)
model = YOLO("yolov8n-seg.pt")

# Load RealSense-captured RGB and depth images
image_path = "/home/queen/test_image.jpg"
depth_image_path = "/home/queen/test_depth.npy"

image = cv2.imread(image_path)
depth_image = np.load(depth_image_path)

# Check if images are loaded correctly
if image is None or depth_image is None:
    print("Error: Could not load image or depth data")
    exit()

# Resize depth image to match RGB image resolution
depth_image = cv2.resize(depth_image, (image.shape[1], image.shape[0]))

# Run YOLOv8 segmentation
results = model(image)

# Process results for bounding boxes & segmentation
for result in results:
    img = result.plot()  # Draw segmentation masks & bounding boxes

    for box, mask in zip(result.boxes.xyxy, result.masks.xy):
        x_min, y_min, x_max, y_max = map(int, box)  # Bounding box coordinates

        # Compute centroid for bounding box
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        # Get real depth at bounding box center (convert mm to meters)
        depth_value_center = depth_image[y_center, x_center] * 0.001

        # Ensure the segmentation mask is properly resized
        mask_resized = np.zeros_like(depth_image)
        cv2.fillPoly(mask_resized, [mask.astype(np.int32)], 1)

        # Extract valid depth pixels inside both bounding box and segmentation mask
        mask_indices = np.where((mask_resized > 0) & (depth_image > 0))

        if len(mask_indices[0]) > 0:
            mask_depths = depth_image[mask_indices] * 0.001
            valid_depths = mask_depths[(mask_depths > 0) & (mask_depths < 10)]  # Remove outliers

            if len(valid_depths) > 0:
                mean_depth = np.mean(valid_depths)  # Compute mean depth for segmentation mask
            else:
                mean_depth = depth_value_center  # Fallback to bounding box depth
        else:
            mean_depth = depth_value_center  # Fallback to bounding box depth

        # Display bounding box depth
        cv2.putText(
            img,
            f"BB Depth: {depth_value_center:.2f}m",
            (x_min, y_min - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Display segmentation depth
        cv2.putText(
            img,
            f"Seg Depth: {mean_depth:.2f}m",
            (x_min, y_min - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

# Show results
cv2.imshow("YOLOv8 Segmentation - Real Depth Measurement", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
