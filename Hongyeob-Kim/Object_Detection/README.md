
### 🌏 재활용 품목 분류를 위한 Object Detection
###### 📌 본 프로젝트는 Naver AI Boostcamp에서 Team Project로 진행됐습니다.

----
### 🍀  최종 결과 
- 2등 (총 21팀)
- private LB : 0.5014

---
### 📝 문제 정의 및 해결 방법
- 해당 대회에 대한 문제를 어떻게 정의하고, 어떻게 풀어갔는지, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서는 [wrapup report](https://maihon.oopy.io/study/boostcamp/p-stage/segmentation-detection/detection-wrapup-report)에서 자세하게 기술하고 있습니다. 
- 위 report에는 대회를 참가한 후, 개인의 회고도 포함되어있습니다. 
- 팀프로젝트를 진행하며 협업 툴로 사용했던 [Notion](https://maihon.oopy.io/a9cad220-042e-4b18-ad92-84b4922eca8d)내용도 해당 링크에 접속하시면 확인하실 수 있습니다.
<br></br>
---
### 💻 CODE 설명
####   - 폴더 구조 

```
├── Object_Detection
    ├── README.md
    ├── input
    │   └── data
    ├── scripts
    ├── src
    └── test_scripts
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
python scripts/experiment_name.sh
```
