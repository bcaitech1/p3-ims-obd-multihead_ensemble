# Pstage 3 ] Semantic Segmentation & Object Detection

## 📋 Table of content

[팀 소개](#Team)<br>
[최종 결과](#Result)<br>
[대회 개요](#Overview)<br>
[문제 정의 해결 및 방법](#Solution)<br>
[CODE 설명](#Code)<br>

<br></br>
## 👋 팀 소개 <a name = 'Team'></a>

- Semantic Segmentation & Object Detection 18조 **Multi-Head Ensemble Team**
- 조원 : 김유지, 김홍엽, 김효진 , 박성배, 박성훈, 오혜린

|                                                                                      김유지                                                                                      |                                                            김홍엽                                                             |                                                          김효진                                                           |                                                            박성배                                                            |                                                            박성훈                                                             |                                                         오혜린                                                             |                                                            
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | 
| <a href='https://github.com/ug-kim'><img src='https://avatars.githubusercontent.com/u/38632805?v=4' width='200px'/></a> | <a href='https://github.com/MaiHon'><img src='https://avatars.githubusercontent.com/u/41847456?v=4' width='200px'/></a> | <a href='https://github.com/vim-hjk'><img src='https://avatars.githubusercontent.com/u/77153072?v=4' width='200px'/></a> | <a href='https://github.com/songbae'><img src='https://avatars.githubusercontent.com/u/65913073?v=4' width='200px'/></a> | <a href='https://github.com/seong0905'><img src='https://avatars.githubusercontent.com/u/70629496?v=4' width='200px'/></a> | <a href='https://github.com/Hyerin-oh'><img src='https://avatars.githubusercontent.com/u/68813518?s=400&u=e5300247dc2b04f5cf57265a6f2e1cc0987e6d08&v=4' width='200px'/></a> 

<br></br>
## 🎖 최종 결과 <a name = 'Result'></a>
- Semantic Segmentaion :
    - private LB : 0.7043 (1등)
    - Public LB :  0.7205 (1등)
    - [1등 발표자료](https://github.com/bcaitech1/p3-ims-obd-multihead_ensemble/blob/master/presentation/Pstage3_solution.pdf) 
- Object Detection :
    - private LB : 0.5014 (2등)
    - Public LB :  0.6068 (3등)

<br></br>
## ♻ 대회 개요 <a name = 'Overview'></a>
환경 부담을 조금이나마 줄일 수 있는 방법의 하나로 '분리수거'가 있습니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립, 소각되기 때문입니다. 우리나라의 분리 수거율은 굉장히 높은 것으로 알려져 있고, 또 최근 이러한 쓰레기 문제가 주목받으며 더욱 많은 사람이 분리수거에 동참하려 하고 있습니다. 하지만 '이 쓰레기가 어디에 속하는지', '어떤 것들을 분리해서 버리는 것이 맞는지' 등 정확한 분리수거 방법을 알기 어렵다는 문제점이 있습니다.

따라서, 우리는 쓰레기가 찍힌 사진에서 쓰레기를 Segmentation 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

- 평가방법 
    - Semantic Segmentation : mAP50 
    - Object Detection : mIoU

<br></br>
## 📝 문제 정의 및 해결 방법 <a name = 'Solution'></a>
- 해당 대회에 대한 문제를 어떻게 정의하고, 어떻게 풀어갔는지, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서는 각자의 wrap up report에서 기술하고 있습니다. 
    - [김유지 wrapup report](https://www.notion.so/Object-Segmentation-798ebd0a47d544bc95148cff5804a600)
    - [김홍엽 wrapup report](https://maihon.oopy.io/study/boostcamp/p-stage/segmentation-detection/wrapup-report)
    - [김효진 wrapup report](https://vimhjk.oopy.io/f31d818e-5128-4a2e-b860-07022002cb48)
    - [박성배 wrapup report](https://songbae.oopy.io/2da200fe-28cf-4c3e-ae7b-f64b312a30dc)
    - [박성훈 wrapup report](https://www.notion.so/Wrap-Up-Report-Stage-3-4ff86742dfb14a4383f620b7fbe13fd1)
    - [오혜린 wrapup report](https://www.notion.so/Wrap-up-Pstage3-Semantic-Segmentation-2679c48f500a40f5bf7d7ffb227b8e46)

- 위 report에는 대회를 참가한 후, 개인의 회고도 포함되어있습니다. 
- 팀프로젝트를 진행하며 협업 툴로 사용했던 [Notion](https://www.notion.so/1cdc0eddd3d649b68eebd94e27dc8655?v=b17e11d3c44148bc80dddf4c24b9cabf)내용도 해당 링크에 접속하시면 확인하실 수 있습니다.

<br></br>
## 💻 CODE 설명<a name = 'Code'></a>
- 자세한 CODE 설명은 개인 폴더 내 README.md를 통해 확인하실 수 있습니다. 
