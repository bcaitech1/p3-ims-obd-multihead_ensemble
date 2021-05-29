# Object Detection and Segmentation

### 재활용품 분류를 위한 Object Detection과 Segmentation
- Semantic Segmentation : `segmentation_models_pytorch`을 활용한 segmentation
- Object Detection : mmdetection을 활용한 detection


### Working Tree 구조

```
.
├── Object_Detection
│   ├── LICENSE
│   ├── README.md
│   ├── README_zh-CN.md
│   ├── configs
│   ├── detectors
│   ├── docker
│   ├── docs
│   ├── faster_rcnn
│   ├── inference.ipynb
│   ├── inference.py
│   ├── mmdet
│   ├── pkl_to_submission.py
│   ├── pytest.ini
│   ├── requirements
│   ├── requirements.txt
│   ├── resources
│   ├── setup.cfg
│   ├── setup.py
│   ├── submission_image
│   ├── temp.py
│   ├── tests
│   ├── tools
│   ├── train.ipynb
│   ├── train.py
│   ├── visualize.py
│   └── yolo
├── README.md
└── Semantic_Segmentation
    ├── README.md
    ├── analysis
    ├── inference.py
    ├── inference_tta.py
    ├── inference_viz.ipynb
    ├── inference_viz_test.ipynb
    ├── requirements.txt
    ├── src
    ├── train.py
    ├── train_pseudo.py
    └── training.ipynb
```