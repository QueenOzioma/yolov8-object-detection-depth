import cv2
import numpy as np
import time
import pyrealsense2 as rs
from ultralytics import YOLO
from hailo_platform import (
    HEF, Device, VDevice, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType, HailoStreamInterface
)

# Load YOLOv8 model for CPU processing
cpu_model = YOLO("yolov8n.pt")  # or "yolov8n-seg.pt" for segmentation

# Load AI Hat Model
devices = Device.scan()
hef_path = "/home/queen/Downloads/yolov8n_seg.hef"
hef = HEF(hef_path)

# RealSense configuration
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from RealSense
pipeline.start(config)

try:
    with VDevice(device_ids=devices) as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32)

        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            while True:
                # Capture frame from RealSense
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                image = np.asanyarray(color_frame.get_data())

                # CPU Inference
                cpu_start = time.time()
                cpu_results = cpu_model(image)
                cpu_end = time.time()
                cpu_fps = 1 / (cpu_end - cpu_start)

                # Draw CPU result
                for result in cpu_results:
                    for box, class_id in zip(result.boxes.xyxy, result.boxes.cls):
                        x_min, y_min, x_max, y_max = map(int, box)
                        class_name = cpu_model.names[int(class_id)]

                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(image, f"{class_name}", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Resize image to fit AI Hat model input
                input_data = cv2.resize(image, (640, 640))
                input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

                # AI Hat Inference
                ai_start = time.time()
                with network_group.activate(network_group_params):
                    infer_pipeline.infer({"yolov8n_seg/input_layer1": input_data})
                ai_end = time.time()
                ai_fps = 1 / (ai_end - ai_start)

                # Overlay FPS values
                cv2.putText(image, f"CPU FPS: {cpu_fps:.2f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"AI Hat FPS: {ai_fps:.2f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display frame
                cv2.imshow("CPU vs AI Hat FPS Comparison", image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

except Exception as e:
    print(f"Error during AI Hat processing: {e}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

