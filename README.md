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

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/macpro_vision_ml.git
cd macpro_vision_ml

# Make setup script executable
chmod +x setup.sh
```

### 2. One-Command Startup

```bash
# Start everything (virtual environment + Jupyter Lab + Label Studio)
source ./setup.sh
```

**⚠️ Important:** Always use `source ./setup.sh`, not `./setup.sh`

### 3. What You Get

After running the setup script:
- ✅ **Virtual environment** activated in your shell
- ✅ **Jupyter Lab**: http://localhost:8888 (for development)
- ✅ **Label Studio**: http://localhost:8080 (for annotation)
- ✅ **All packages** installed and verified

## 📋 Environment Management

### Available Commands

After running `source ./setup.sh`, you have these commands:

```bash
check_services    # Check what's currently running
stop_services     # Stop both Jupyter Lab and Label Studio
show_help         # Show detailed help and troubleshooting
```

### Daily Workflow

```bash
# Morning: Start everything
source ./setup.sh

# Work all day using:
# - Jupyter Lab at localhost:8888
# - Label Studio at localhost:8080
# - Virtual environment active in your shell

# Evening: Stop everything
stop_services
```

### Troubleshooting

```bash
# Check what's running
check_services

# View error logs
tail label_studio.log
tail jupyter.log

# Manual cleanup if needed
pkill -f jupyter
pkill -f label-studio

# Get help
show_help
```

## 📁 Project Structure

```
macpro_vision_ml/
├── setup.sh                    # One-command startup script
├── requirements.txt             # Package dependencies
├── .venv/                      # Virtual environment (auto-created)
├── 1_vision_model_setup.ipynb  # Vision model fundamentals
├── 2_label_studio_setup.ipynb  # Annotation setup
├── 3_data_pipeline.ipynb       # Data conversion pipeline
├── projects/                   # Your client projects
│   └── your_project_name/
│       ├── raw_images/
│       └── project-export.json
└── models/                     # Trained models
```

## 📋 Complete Workflow Guide

### Phase 1: Environment Setup

#### Step 1: Initial Setup (One Time)
```bash
source ./setup.sh
```

This automatically:
- Creates/activates virtual environment
- Installs all required packages
- Starts Jupyter Lab and Label Studio
- Verifies everything works

#### Step 2: Navigate to Notebooks
Open http://localhost:8888 and run notebooks in sequence:
1. `1_vision_model_setup.ipynb` - Vision model fundamentals
2. `2_label_studio_setup.ipynb` - Annotation project setup
3. `3_data_pipeline.ipynb` - Data conversion pipeline

### Phase 2: Project Setup

#### Step 1: Create Project Structure
```bash
mkdir projects/your_project_name
cd projects/your_project_name
mkdir raw_images
```

#### Step 2: Organize Your Images
```
projects/your_project_name/
├── raw_images/              # Place your source images here
├── project-export.json      # Label Studio export (after annotation)
└── README.md               # Project-specific documentation
```

### Phase 3: Data Annotation with Label Studio

#### Step 1: Access Label Studio
Open http://localhost:8080 (automatically started by setup script)

#### Step 2: Create Project
1. Click "Create Project"
2. Name your project (e.g., "Roof Damage Detection")
3. **Data Import**: Upload images from your `raw_images/` folder
4. **Labeling Setup**: Choose "Object Detection with Bounding Boxes"

#### Step 3: Configure Labels
```xml
<View>
  <Header value="Damage Assessment"/>
  <Image name="image" value="$image" zoom="true"/>
  <RectangleLabels name="damage" toName="image">
    <Label value="no_damage" background="#00ff00"/>
    <Label value="light_damage" background="#ffff00"/>
    <Label value="moderate_damage" background="#ff8800"/>
    <Label value="severe_damage" background="#ff0000"/>
  </RectangleLabels>
</View>
```

#### Step 4: Annotation Process
1. Select damage type (colored button)
2. Click and drag to draw bounding box around damage
3. Repeat for all damage areas in the image
4. Click "Submit" to save annotation

#### Step 5: Export Data
1. Go to project dashboard
2. Click "Export" → "JSON"
3. Save as `project-export.json` in your project directory

### Phase 4: Data Pipeline Conversion

#### Step 1: Convert Annotations
In Jupyter Lab, run `3_data_pipeline.ipynb`:

```python
# Convert Label Studio export to YOLO training format
convert_roof_damage_dataset(
    'projects/your_project/project-export.json',  # Label Studio export
    'projects/your_project/raw_images/',          # Source images  
    'training_dataset_v1',                        # Output dataset name
    split_ratio=0.8                               # 80% train, 20% val
)
```

#### Step 2: Verify Dataset
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

### Phase 5: Model Training

#### Step 1: Train Object Detection Model
```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')  # Nano model for M4 Pro

# Train on your dataset
results = model.train(
    data='training_dataset_v1/data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,                    # Optimized for M4 Pro
    device='mps',              # Apple Silicon GPU
    project='models',
    name='your_project_v1'
)
```

#### Step 2: Expected Performance

| Dataset Size | Training Time | Expected mAP50 |
|-------------|---------------|----------------|
| 10-50 images | 5-20 minutes | 0.3-0.6 (proof of concept) |
| 100-500 images | 30-60 minutes | 0.6-0.8 (good performance) |
| 1000+ images | 2-4 hours | 0.8+ (production ready) |

### Phase 6: Model Testing and Demo

#### Step 1: Load Trained Model
```python
# Load your trained model
trained_model = YOLO('models/your_project_v1/weights/best.pt')

# Test on new image
results = trained_model.predict(
    'test_image.jpg',
    conf=0.25,                 # Confidence threshold
    save=True                  # Save annotated result
)
```

#### Step 2: Create Demo Function
```python
def create_demo_function(model_path, class_names):
    demo_model = YOLO(model_path)
    
    def detect_damage(image_path, confidence=0.25):
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
            'damage_found': len(detections) > 0
        }
    
    return detect_damage

# Create demo function
demo = create_demo_function(
    'models/your_project_v1/weights/best.pt',
    ['no_damage', 'light_damage', 'moderate_damage', 'severe_damage']
)

# Use with client images
result = demo('client_image.jpg')
print(f"Found {result['total_objects']} damage areas")
```

## 🔧 Advanced Configuration

### Custom Training for Small Datasets (<100 images)
```python
model.train(
    data='data.yaml',
    epochs=200,                     # More epochs
    batch=4,                        # Smaller batch
    imgsz=416,                      # Smaller image size
    lr0=0.001,                      # Lower learning rate
    augment=True,                   # Heavy augmentation
)
```

### Custom Training for Large Datasets (1000+ images)
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
model = YOLO('models/your_project_v1/weights/best.pt')

# Export to ONNX (cross-platform)
model.export(format='onnx')

# Export to CoreML (iOS/macOS)
model.export(format='coreml')

# Export to TensorFlow
model.export(format='tf')
```

## 🚨 Troubleshooting

### Environment Issues
```bash
# Check environment status
check_services

# If virtual environment not active
source .venv/bin/activate

# If services won't start
stop_services
pkill -f jupyter
pkill -f label-studio
source ./setup.sh

# If packages missing
pip install -r requirements.txt
```

### Training Issues
```bash
# If training crashes with memory error
# Reduce batch size: batch=4 or batch=1

# If MPS not available
python -c "import torch; print(torch.backends.mps.is_available())"
# Should return True on M4 Pro with macOS 12.3+

# If no objects detected after training
# Lower confidence threshold: conf=0.01
# Check dataset labels format
```

### Label Studio Issues
```bash
# If Label Studio won't load
# Check: http://localhost:8080
# Try restarting: stop_services && source ./setup.sh

# If images won't display
# Check file paths and permissions
# Ensure images are in correct directory
```

## 📈 Project Templates

### Template 1: Damage Detection
```
Classes: no_damage, light_damage, moderate_damage, severe_damage
Use Case: Insurance claims, maintenance inspection
Typical Dataset: 100-500 images per class
```

### Template 2: Quality Control
```
Classes: defect_free, minor_defect, major_defect, critical_defect
Use Case: Manufacturing quality inspection
Typical Dataset: 200-1000 images per class
```

### Template 3: Security Monitoring
```
Classes: person, vehicle, package, intruder
Use Case: Automated security systems
Typical Dataset: 500-2000 images per class
```

## 🤝 Client Handoff

### Deliverables Checklist
- [ ] Trained model files (`best.pt`)
- [ ] Training performance report
- [ ] Demo inference function
- [ ] Deployment documentation
- [ ] Data collection guidelines
- [ ] Model update process

### Model Versioning
```
models/
├── client_project_v1.0/
│   ├── weights/best.pt
│   ├── training_config.yaml
│   └── performance_report.pdf
├── client_project_v2.0/
│   ├── weights/best.pt
│   ├── training_config.yaml
│   └── performance_report.pdf
└── latest/                     # Symlink to current best
```

## 📚 Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Label Studio Documentation](https://labelstud.io/guide/)
- [PyTorch MPS Guide](https://pytorch.org/docs/stable/notes/mps.html)

## 📄 License

MIT License - free for commercial use and client projects.

---

**🚀 Professional computer vision pipeline ready for MacBook Pro M4!**

**Quick Commands:**
- Start: `source ./setup.sh`
- Check: `check_services`
- Stop: `stop_services`
- Help: `show_help`