## ♻️ 재활용 품목 분류를 위한 Semantic Segmentation & Object Detection

###### 📌 본 프로젝트는 [_*Naver AI Boostcamp*_](https://www.edwith.org/bcaitech1/)에서 Team Project로 진행됐습니다.
<br>


----
### 🍀  최종 결과 
- [[Semantic Segmentation]](http://boostcamp.stages.ai/competitions/28/overview/description)
    - **1등 (총 21팀)**
    - public  LB : 0.7205 
    - private LB : 0.7043
    - [[발표 자료]](https://www.notion.so/MultiHead_Ensemble-a6d4e3db725a4588ab18ab7ea2551c92#0ace36d4004d4f17913cc543888fa0bd)
    - [[Code]](https://github.com/bcaitech1/p3-ims-obd-multihead_ensemble/blob/master/Hongyeob-Kim/Semantic_Segmentation/)
<br></br>

- [[Object Detection]](http://boostcamp.stages.ai/competitions/35/overview/description)
    - 2등 (총 21팀)
    - Public  LB : 0.6068
    - private LB : 0.5014
    - [[Code]](./Object_Detection/)
---
### 🗄 폴더 구조
```
├── Obejct_Detection
∣    ├── scripts
∣    ├── src
∣    └── test_scripts
∣
└── Semantic_Segmentation
     ├── experiments
     ├── src
     ∣    ├── configs
     ∣    ├── models
     ∣    ├── augmix.py         
     ∣    ├── dataset.py        
     ∣    ├── losses.py         
     ∣    ├── schedulers.py     
     ∣    ├── utils.py          
     ∣    └── warping.py        
     ∣
     ├── test_scripts
     ├── augmix_train.py
     ├── ensemble_test.py
     ├── pseudo_train.py
     ├── ensemble_test.py
     ├── train_eval.py
     ├── tta_ensemble_test.py
     └── tta_test.py
```
