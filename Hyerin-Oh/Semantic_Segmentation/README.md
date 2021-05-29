
### 🌏 재활용 품목 분류를 위한 Semantic Segmentation
<br>

###### 📌 본 프로젝트는 Naver AI Boostcamp에서 Team Project로 진행됐습니다.

----
### 🍀  최종 결과 
- 1등 (총 21팀)
- private LB : 0.7043
- [1등 발표 자료](https://drive.google.com/file/d/1gXRMAgSluj0UkybFLYOQMOAFLcrYsAAs/view?usp=sharing)는 여기서 확인하실 수 있습니다. 

---
### 📝 문제 정의 및 해결 방법
- 해당 대회에 대한 문제를 어떻게 정의하고, 어떻게 풀어갔는지, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서는 [wrapup report](https://www.notion.so/Wrap-up-Pstage3-Semantic-Segmentation-2679c48f500a40f5bf7d7ffb227b8e46)에서 자세하게 기술하고 있습니다. 
- 위 report에는 대회를 참가한 후, 개인의 회고도 포함되어있습니다. 
- 팀프로젝트를 진행하며 협업 툴로 사용했던 [Notion ](https://www.notion.so/1cdc0eddd3d649b68eebd94e27dc8655?v=b17e11d3c44148bc80dddf4c24b9cabf)내용도 해당 링크에 접속하시면 확인하실 수 있습니다.
<br>
---
### 💻 CODE 설명
####   - 폴더 구조 
<br>

```
├── config                  # 실험 config 코드
∣    ├── config.yml             # train   
|    └── eval_config.yml        # infernece 
|
├── src                     # source 코드
|    ├── dataset                
|    ├── losses                 
|    ├── scheduler                             
|    ├── train              # 학습
|    ├── add_train          # pseudo data를 이용해서 train할 때
|    ├── eval               # 추론
|    └── utils              # 그 외 
└── main
main.py
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
python main.py --config_path "config가 있는 파일 경로"--config "본인이 실험하고 싶은 config 파일 내 이름" --run_name "wandb 사용 시 실험 이름"
```