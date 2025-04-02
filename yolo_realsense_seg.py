import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 Segmentation model
model = YOLO("yolov8n-seg.pt")  # Use segmentation model

# Configure RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Get depth scale
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Align depth to color
align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert to numpy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale  # Convert mm to meters

        # Run YOLOv8 Segmentation
        results = model(color_image)

        for result in results:
            img = result.plot()  # Draw masks and bounding boxes

            # Ensure `masks` exist before processing
            if result.masks is None:
                continue  # Skip if no segmentation masks are detected

            # Extract masks & bounding boxes
            for box, mask in zip(result.boxes.xyxy, result.masks.xy):
                x_min, y_min, x_max, y_max = map(int, box)

                # Compute bounding box center
                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2

                # Get depth at bounding box center
                bb_depth = depth_frame.get_distance(x_center, y_center)

                # Resize mask to match depth image resolution
                mask_resized = cv2.resize(mask.astype(np.uint8), (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_indices = np.where(mask_resized > 0)

                # Extract depth values inside mask and remove invalid depths
                mask_depths = depth_image[mask_indices]
                valid_depths = mask_depths[(mask_depths > 0) & (mask_depths < 10)]  # Remove invalid depth values

                # Compute mean depth from segmentation mask
                seg_depth = np.mean(valid_depths) if valid_depths.size > 0 else bb_depth

                # Display depth values
                cv2.putText(img, f"BB Depth: {bb_depth:.2f}m", (x_min, y_min - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(img, f"Seg Depth: {seg_depth:.2f}m", (x_min, y_min - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                print(f"Object at ({x_center}, {y_center}): BB Depth = {bb_depth:.2f}m, Seg Depth = {seg_depth:.2f}m")

            cv2.imshow("YOLOv8-Seg RealSense", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
