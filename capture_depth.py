import pyrealsense2 as rs
import numpy as np
import cv2

#  Configure RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

#  Start streaming
pipeline.start(config)

try:
    print("Capturing Depth Image... Press 's' to save or 'q' to quit.")
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        #  Convert frames to NumPy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Normalize depth image for display (optional)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        #  Display color and depth images
        combined = np.hstack((color_image, depth_colormap))
        cv2.imshow("RealSense Depth Capture", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to save
            np.save("/home/queen/test_depth.npy", depth_image)  # Save depth image
            cv2.imwrite("/home/queen/test_image.jpg", color_image)  # Save RGB image
            print(" Depth image and RGB image saved successfully!")
            break
        elif key == ord('q'):  # Press 'q' to quit without saving
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
