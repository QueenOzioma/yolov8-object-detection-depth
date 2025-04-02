# **Object Detection, Segmentation and Distance Estimate using YOLOv8**

## ** Project Overview**
This project compares **YOLOv8 inference speed** FPS between:
- **CPU-based processing** (using OpenCV & PyTorch)
- **AI Hat hardware acceleration** (Hailo AI Hat on Raspberry Pi 5)

It also evaluates **depth measurement accuracy** using two depth measurement methods:
1. **Bounding Box Depth**: Uses the depth at the center of the detected object.
2. **Segmentation Mask Depth**: Averages depth values across the detected region.

---

## **Repository Structure**

 /fps_comparison.py # Compares CPU and AI Hat FPS depth_measurement.py # Computes depth using bounding boxes & segmentation 
 /images test_image.jpg # Sample test image used for object detection 
 /depth_data test_depth.npy # Corresponding depth data file for test images README.md # Project documentation


---

# **Running FPS Comparison**
This script compares inference speed (FPS) between **CPU and AI Hat**.

```bash


 Expected Output

    FPS Comparison Output
        Bounding boxes for detected objects.
        FPS for both CPU and AI Hat.

    Depth Measurement Output
        Bounding box and segmentation depth values displayed on the image.
**# **CPU_yolo_realsense_seg.py****
This script evaluates two depth measurement methods- bounding box and segmentation
