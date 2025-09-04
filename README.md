# Research on Multi-Modal Visual Data Fusion Techniques

## 📌 Overview
This project investigates techniques for fusing **2D RGB images** with **3D LiDAR point cloud data** to:
- Estimate distances from the camera to objects.
- Improve object detection performance.

We utilize **YOLOv8 (Ultralytics)** and **Open3D** to implement and compare different fusion strategies, and provide an **interactive demo using Streamlit**.

---

## 📂 Dataset
The original KITTI dataset is **not included** in the repository due to its large size. Please download manually from [KITTI Object Detection benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d):

- **Left color images of object data set** (12 GB) → `image_2/` (RGB images)  
- **Velodyne point clouds** (29 GB) → `data_object_velodyne/`  
- **Camera calibration matrices** (16 MB) → `data_object_calib/`  
- **Training labels** (5 MB) → `data_object_label_2/`  

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Hate0205/Research-on-techniques-that-combine-multiple-visual-data.git
cd Research-on-techniques-that-combine-multiple-visual-data
