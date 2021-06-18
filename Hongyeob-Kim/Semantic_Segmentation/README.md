
### 🌏 재활용 품목 분류를 위한 Semantic Segmentation
###### 📌 본 프로젝트는 Naver AI Boostcamp에서 Team Project로 진행됐습니다.

----
### 🍀  최종 결과 
- 1등 (총 21팀)
- private LB : 0.7043
- [1등 발표 자료](https://github.com/bcaitech1/p3-ims-obd-multihead_ensemble/blob/master/presentation/Pstage3_solution.pdf)는 여기서 확인하실 수 있습니다. 

---
### 📝 문제 정의 및 해결 방법
- 해당 대회에 대한 문제를 어떻게 정의하고, 어떻게 풀어갔는지, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서는 [wrapup report](https://maihon.oopy.io/study/boostcamp/p-stage/segmentation-detection/segmentation-wrapup-report)에서 자세하게 기술하고 있습니다. 
- 위 report에는 대회를 참가한 후, 개인의 회고도 포함되어있습니다. 
- 팀프로젝트를 진행하며 협업 툴로 사용했던 [Notion ](https://www.notion.so/1cdc0eddd3d649b68eebd94e27dc8655?v=b17e11d3c44148bc80dddf4c24b9cabf)내용도 해당 링크에 접속하시면 확인하실 수 있습니다.
<br></br>
---
### 💻 CODE 설명
####   - 폴더 구조 

```
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

<br></br>

####   - 소스 설명 
- `train_eval.py` : 기본 train & validation 코드
- `pseudo_train.py` : train데이터에 단순히 pseudo labeling한 데이터를 더하여 학습 코드
- `scheduler.py` : cosine annealing with warm starts를 사용
- `losses.py` : 실험해본 loss 코드
- `augmix.py` : Songbaemix 코드

<br></br>
#### - 실행하는 법
``` 
python experiments/experiment_name.sh
```
