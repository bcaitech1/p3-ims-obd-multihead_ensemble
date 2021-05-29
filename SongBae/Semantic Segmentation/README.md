
### 🌏 재활용 품목 분류를 위한 Semantic Segmentation
<br>

###### 📌 본 프로젝트는 Naver AI Boostcamp에서 Team Project로 진행됐습니다.
###### 📆 2021.04.26~2021.05.21

----
### 🏆  최종 결과 
- 1등 (총 21팀)
- private LB : 0.7043
- [1등 발표 자료](https://drive.google.com/file/d/1gXRMAgSluj0UkybFLYOQMOAFLcrYsAAs/view?usp=sharing)는 여기서 확인하실 수 있습니다. 

---
### 📝 문제 정의 및 해결 방법
- 해당 대회에 대한 문제를 어떻게 접근하고, 풀어갔고, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서 [wrapup report](https://songbae.oopy.io/2da200fe-28cf-4c3e-ae7b-f64b312a30dc)에서 자세하게 기술하고 있습니다. 


- 팀프로젝트의 전반적인 내용은 [Notion ](https://songbae.oopy.io/3b39510e-4932-4e87-82e5-5c1f61d724fd) 에서 확인가능합니다
<br>
---
### 💻 CODE 설명
####   - 폴더 구조 
<br>

```
├── config                  # 실험 config 코드
∣    ├── hrnet_seg.yaml             # train   
|           
|
├── src                     # source 코드
|    ├── dataset            # dataset     
|    ├── losses             # loss function 정의
|    ├── models             # 사용 모델              
|    ├── utils              # 그 외 필요한 function
|    
|    
├── test.py                 # inference 코드    
└── train.py                # train 코드  
main
```

<br>

####   - 소스 설명 
- `datset.py` : train / val dataset 생성 (object aug를 사용할 지 선택 가능)
- `losses.py` : semantic segmentation loss 모아놓은 코드 , import module을 통해 불러와서 train시 사용
- `scheduler.py` : cosine annealing with warm starts를 사용
- `train.py` : train dataset만을 학습할 시 사용
- `add_train.py` : train dataset과 pseudo dataset을 학습할 때 사용
- `eval.py` : 추론 시 사용
- `utils.py` : 그 외 모든 기능 (ex. Dataloader , CRF , Cutout...)

<br>

``` 
python train.py --cfg #config args를 통해 수정

python test.py --cfg # config args를 통해 수정
```