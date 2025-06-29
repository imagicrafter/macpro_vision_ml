# 🏠 MacBook Pro M4 Vision ML Pipeline

A complete computer vision machine learning pipeline optimized for MacBook Pro M4, featuring Label Studio annotation, YOLOv8 training, and professional client demonstrations.

## 🎯 Overview

This repository provides a complete workflow for:

- **Data Annotation** using Label Studio (self-hosted)
- **Object Detection Training** with YOLOv8 on Apple Silicon
- **Model Inference** and client demonstrations
- **Professional ML Pipeline** for consulting projects

## 🔧 System Requirements

| Component | Requirement | Tested Configuration |
|-----------|-------------|---------------------|
| **Hardware** | MacBook Pro M4/M4 Pro | ✅ M4 Pro, 24GB RAM |
| **macOS** | 12.3+ (for MPS support) | ✅ Latest macOS |
| **Python** | 3.12+ | ✅ 3.12.9 |
| **Memory** | 8GB+ RAM | ✅ 24GB unified memory |

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/imagicrafter/macpro_vision_ml.git
cd macpro_vision_ml

# Create Python 3.12 virtual environment
python3.12 -m venv ml_env
source ml_env/bin/activate

# Install core dependencies
pip install jupyter jupyterlab ultralytics label-studio
pip install torch torchvision matplotlib opencv-python pillow
pip install pandas numpy scikit-learn
```

### 2. Start JupyterLab

```bash
# From project root directory
jupyter lab
```

### 3. Follow the Notebook Sequence

1. **Setup & Classification**: `1_vision_model_setup.ipynb`
2. **Data Annotation**: `2_label_studio_setup.ipynb` 
3. **Data Pipeline**: `3_data_pipeline.ipynb`
4. **Object Detection Training**: `4_object_detection_training.ipynb`

## 📋 Complete Workflow Guide

### Phase 1: Project Setup

#### Step 1: Create Project Structure
```bash
mkdir projects/your_project_name
cd projects/your_project_name
```

#### Step 2: Organize Your Images
```
projects/your_project_name/
├── raw_images/              # Place your source images here
├── project-export.json      # Label Studio export (after annotation)
└── README.md               # Project-specific documentation
```

### Phase 2: Data Annotation with Label Studio

#### Step 1: Setup Label Studio
Run notebook: `2_label_studio_setup.ipynb`

```python
# Creates annotation project structure
setup_label_studio_project()
```

This creates:
```
your_project_annotation_project/
├── raw_images/              # Upload your images here
├── exported_data/           # Label Studio exports
├── labeling_config.xml      # Annotation configuration
└── README.md               # Setup instructions
```

#### Step 2: Start Label Studio Server
```bash
cd your_project_annotation_project
label-studio start --port 8080
```

#### Step 3: Configure Project
1. Open http://localhost:8080
2. Create new project: "Your Project Name"
3. **Data Import**: Upload images from `raw_images/`
4. **Labeling Setup**: Copy configuration from `labeling_config.xml`

#### Step 4: Annotation Guidelines

**For Object Detection (Bounding Boxes):**
```xml
<RectangleLabels name="damage_detection" toName="image">
    <Label value="class_1" background="#00ff00"/>
    <Label value="class_2" background="#ffff00"/>
    <Label value="class_3" background="#ff8800"/>
    <Label value="class_4" background="#ff0000"/>
</RectangleLabels>
```

**Annotation Process:**
1. Select damage type (colored button)
2. Click and drag to draw bounding box
3. Repeat for all damage areas
4. Submit annotation

#### Step 5: Export Annotations
1. In Label Studio: **Export** → **JSON** 
2. Save as `project-export.json` in your project directory

### Phase 3: Data Pipeline Conversion

#### Step 1: Run Data Pipeline
Notebook: `3_data_pipeline.ipynb`

```python
# Convert Label Studio export to training format
convert_roof_damage_dataset(
    'project-export.json',           # Label Studio export
    'raw_images/',                   # Source images  
    'training_dataset_v1',           # Output dataset name
    split_method='ratio',            # or 'manual'
    split_ratio=0.8                  # 80% train, 20% val
)
```

#### Step 2: Verify Dataset Structure
```
training_dataset_v1/
├── images/
│   ├── train/                   # Training images
│   └── val/                     # Validation images
├── labels/
│   ├── train/                   # YOLO format labels (.txt)
│   └── val/                     # YOLO format labels (.txt)
├── visualizations/              # Annotation previews
├── data.yaml                    # YOLO configuration
└── dataset_info.json           # Conversion statistics
```

### Phase 4: Model Training

#### Step 1: Object Detection Training
Notebook: `4_object_detection_training.ipynb`

#### Step 2: Training Configuration
```python
# Optimized for M4 Pro
training_config = {
    'model_size': 'yolov8n.pt',     # Nano for speed
    'epochs': 100,                   # More epochs for small datasets
    'batch_size': 8,                # Conservative for M4 Pro  
    'image_size': 640,              # Standard YOLO input
    'device': 'mps',                # Apple Silicon GPU
    'patience': 20,                 # Early stopping
}
```

#### Step 3: Start Training
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='training_dataset_v1/data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    device='mps',
    project='model_training',
    name='experiment_v1'
)
```

#### Step 4: Expected Performance

| Dataset Size | Training Time | Expected mAP50 |
|-------------|---------------|----------------|
| 10-50 images | 5-20 minutes | 0.3-0.6 (proof of concept) |
| 100-500 images | 30-60 minutes | 0.6-0.8 (good performance) |
| 1000+ images | 2-4 hours | 0.8+ (production ready) |

### Phase 5: Model Evaluation and Inference

#### Step 1: Analyze Training Results
```python
# Validation metrics
val_results = model.val()
print(f"mAP50: {val_results.box.map50:.3f}")
print(f"mAP50-95: {val_results.box.map:.3f}")
```

#### Step 2: Test Inference
```python
# Load trained model
trained_model = YOLO('model_training/experiment_v1/weights/best.pt')

# Test on new image
results = trained_model.predict(
    'test_image.jpg',
    conf=0.25,                      # Confidence threshold
    save=True                       # Save annotated image
)
```

#### Step 3: Create Demo Function
```python
def create_demo_function(model_path, class_names):
    demo_model = YOLO(model_path)
    
    def detect_objects(image_path, confidence=0.25):
        results = demo_model.predict(image_path, conf=confidence)
        
        detections = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                detection = {
                    'class': class_names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)
        
        return {
            'detections': detections,
            'total_objects': len(detections),
            'objects_found': len(detections) > 0
        }
    
    return detect_objects

# Create demo function
demo_function = create_demo_function(
    'model_training/experiment_v1/weights/best.pt',
    ['class_1', 'class_2', 'class_3', 'class_4']
)

# Use demo function
result = demo_function('client_image.jpg')
print(f"Found {result['total_objects']} objects")
```

## 📊 Project Templates

### Template 1: Damage Detection
```
Project: Building/Infrastructure Damage
Classes: no_damage, light_damage, moderate_damage, severe_damage
Use Case: Insurance claims, maintenance planning
```

### Template 2: Quality Control
```
Project: Manufacturing Quality Control  
Classes: defect_free, minor_defect, major_defect, critical_defect
Use Case: Automated quality inspection
```

### Template 3: Medical Imaging
```
Project: Medical Image Analysis
Classes: normal, abnormal, requires_attention
Use Case: Screening and diagnosis support
```

## 🔧 Advanced Configuration

### Custom Split Strategies

#### Manual Split (Control which images go to validation)
```python
convert_custom_dataset(
    'export.json',
    'images/',
    'dataset_manual_split',
    split_method='manual',
    train_indices=[0, 1, 2, 3, 4, 5]  # First 6 for training
)
```

#### Balanced Split (Even distribution across classes)
```python
convert_custom_dataset(
    'export.json', 
    'images/',
    'dataset_balanced',
    split_ratio=0.7  # 70/30 split
)
```

### Training Optimizations

#### For Small Datasets (<100 images)
```python
model.train(
    data='data.yaml',
    epochs=200,                     # More epochs
    batch=1,                        # Smaller batch
    imgsz=416,                      # Smaller image size
    lr0=0.001,                      # Lower learning rate
    hsv_h=0.1, hsv_s=0.9, hsv_v=0.9, # Heavy augmentation
    degrees=30, translate=0.2,      # More augmentation
    mosaic=1.0, mixup=0.5          # Data mixing
)
```

#### For Large Datasets (1000+ images)
```python
model.train(
    data='data.yaml',
    epochs=50,                      # Fewer epochs needed
    batch=16,                       # Larger batch size
    imgsz=640,                      # Full resolution
    lr0=0.01,                       # Standard learning rate
    device='mps'                    # Apple Silicon acceleration
)
```

## 📤 Model Export and Deployment

### Export for Different Platforms
```python
# Load trained model
model = YOLO('path/to/best.pt')

# Export to ONNX (cross-platform)
model.export(format='onnx')

# Export to CoreML (iOS/macOS)
model.export(format='coreml')

# Export to TensorFlow
model.export(format='tf')
```

### Client Deployment Options

#### Option 1: Python API
```python
from ultralytics import YOLO

class ObjectDetectionAPI:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def predict(self, image_path):
        return self.model.predict(image_path)
```

#### Option 2: Web Service
```python
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle image upload and prediction
    pass
```

#### Option 3: iOS/macOS App
- Use exported CoreML model
- Integrate with native Swift/Objective-C
- Real-time camera inference

## 🚨 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# If ultralytics installation fails
pip install ultralytics --no-cache-dir

# If torch MPS not available
# Check macOS version (requires 12.3+)
python -c "import torch; print(torch.backends.mps.is_available())"
```

#### Training Issues
```bash
# If training crashes with memory error
# Reduce batch size in training config
batch=4  # or batch=1 for very limited memory

# If no objects detected after training
# Check dataset format and labels
# Lower confidence threshold: conf=0.01
```

#### Label Studio Issues
```bash
# If Label Studio won't start
label-studio start --port 8081  # Try different port

# If images won't load
# Check file permissions and paths
# Ensure images are in correct directory
```

### Performance Optimization

#### For Faster Training
- Use `yolov8n.pt` (nano model)
- Reduce image size: `imgsz=416`
- Increase batch size if memory allows
- Use MPS acceleration: `device='mps'`

#### For Better Accuracy  
- Use `yolov8s.pt` or `yolov8m.pt`
- Increase epochs for small datasets
- Add more training data
- Use data augmentation

## 📈 Scaling for Production

### Data Collection Strategy
1. **Start small**: 10-50 images for proof of concept
2. **Iterative improvement**: Add 100-200 images per iteration  
3. **Production ready**: 1000+ images with good class balance
4. **Continuous learning**: Regular model updates with new data

### Model Versioning
```
models/
├── v1.0_proof_of_concept/
│   ├── best.pt
│   ├── training_config.yaml
│   └── performance_metrics.json
├── v2.0_production/
│   ├── best.pt  
│   ├── training_config.yaml
│   └── performance_metrics.json
└── latest/                     # Symlink to current best model
```

### Client Handoff Checklist
- [ ] Trained model files (`best.pt`)
- [ ] Training performance report
- [ ] Demo inference function
- [ ] Deployment documentation
- [ ] Data collection guidelines for improvement
- [ ] Model update/retraining process

## 📚 Resources

### Documentation
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Label Studio Documentation](https://labelstud.io/guide/)
- [PyTorch MPS Guide](https://pytorch.org/docs/stable/notes/mps.html)

### Example Projects
- Roof damage detection
- Manufacturing quality control
- Medical image analysis
- Retail inventory management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test on MacBook Pro M4 setup
4. Submit pull request with performance benchmarks

## 📄 License

MIT License - feel free to use for commercial projects and client work.

---

**🚀 Ready to build professional computer vision solutions on MacBook Pro M4!**