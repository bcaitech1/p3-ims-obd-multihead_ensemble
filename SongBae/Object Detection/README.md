
### 🌏 재활용 품목 분류를 위한 Object Detection
<br>

###### 📌 본 프로젝트는 Naver AI Boostcamp에서 Team Project로 진행됐습니다.
###### 📆 2021.04.26~2021.05.21

----
### 🏆  최종 결과 
- 2등 (총 21팀)
- private LB : 0.5043


---
### 📝 문제 정의 및 해결 방법
- 해당 대회에 대한 문제를 어떻게 접근하고, 풀어갔고, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서 [wrapup report](https://songbae.oopy.io/a6214749-5886-4f21-992d-4e11f5660028)에서 자세하게 기술하고 있습니다. 


- 팀프로젝트의 전반적인 내용은 [Notion ](https://songbae.oopy.io/e7a84cf1-0ad6-4186-87d4-e0571145fa29) 에서 확인가능합니다
<br>
---
### 💻 CODE 설명
####   - 폴더 구조 
<br>

```
├── configs                  # 실험 config 코드
∣    ├── custom_swin_base            # train   
|           
|
├── src                     # source 코드
|    ├── dataset            # dataset  
|    ├── ensemble           # 앙상블 코드 
|               
|    
├── test.py                 # inference 코드    
└── train.py                # train 코드  
main
```

<br>

####   - 소스 설명 
- `datset.py` : train / val dataset 생성
- `ensemble.py` : inference시 앙상블 사용
- `train.py` : train dataset을 학습할 시 사용
- `test.py` : 추론 시 사용

<br>

``` 
python train.py --cfg #config args를 통해 수정

python test.py --cfg # config args를 통해 수정
```