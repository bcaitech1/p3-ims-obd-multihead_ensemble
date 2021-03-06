# Pstage 3 ] Semantic Segmentation & Object Detection

## ๐ Table of content

- [ํ ์๊ฐ](#Team)<br>
- [์ต์ข ๊ฒฐ๊ณผ](#Result)<br>
- [๋ํ ๊ฐ์](#Overview)<br>
- [๋ฌธ์  ์ ์ ํด๊ฒฐ ๋ฐ ๋ฐฉ๋ฒ](#Solution)<br>
- [CODE ์ค๋ช](#Code)<br>

<br></br>
## ๐ ํ ์๊ฐ <a name = 'Team'></a>

- Semantic Segmentation & Object Detection 18์กฐ **Multi-Head Ensemble Team**
- ์กฐ์ : ๊น์ ์ง, ๊นํ์ฝ, ๊นํจ์ง , ๋ฐ์ฑ๋ฐฐ, ๋ฐ์ฑํ, ์คํ๋ฆฐ

|                                                                                      ๊น์ ์ง                                                                                      |                                                            ๊นํ์ฝ                                                             |                                                          ๊นํจ์ง                                                           |                                                            ๋ฐ์ฑ๋ฐฐ                                                            |                                                            ๋ฐ์ฑํ                                                             |                                                         ์คํ๋ฆฐ                                                             |                                                            
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | 
| <a href='https://github.com/ug-kim'><img src='https://avatars.githubusercontent.com/u/38632805?v=4' width='200px'/></a> | <a href='https://github.com/MaiHon'><img src='https://avatars.githubusercontent.com/u/41847456?v=4' width='200px'/></a> | <a href='https://github.com/vim-hjk'><img src='https://avatars.githubusercontent.com/u/77153072?v=4' width='200px'/></a> | <a href='https://github.com/songbae'><img src='https://avatars.githubusercontent.com/u/65913073?v=4' width='200px'/></a> | <a href='https://github.com/seong0905'><img src='https://avatars.githubusercontent.com/u/70629496?v=4' width='200px'/></a> | <a href='https://github.com/Hyerin-oh'><img src='https://avatars.githubusercontent.com/u/68813518?s=400&u=e5300247dc2b04f5cf57265a6f2e1cc0987e6d08&v=4' width='200px'/></a> 

<br></br>
## ๐ ์ต์ข ๊ฒฐ๊ณผ <a name = 'Result'></a>
- Semantic Segmentaion :
    - private LB : 0.7043 (1๋ฑ)
    - Public LB :  0.7205 (1๋ฑ)
    - [1๋ฑ ๋ฐํ์๋ฃ](https://github.com/bcaitech1/p3-ims-obd-multihead_ensemble/blob/master/presentation/Pstage3_solution.pdf) 
- Object Detection :
    - private LB : 0.5014 (2๋ฑ)
    - Public LB :  0.6068 (3๋ฑ)

<br></br>
## โป ๋ํ ๊ฐ์ <a name = 'Overview'></a>
ํ๊ฒฝ ๋ถ๋ด์ ์กฐ๊ธ์ด๋๋ง ์ค์ผ ์ ์๋ ๋ฐฉ๋ฒ์ ํ๋๋ก '๋ถ๋ฆฌ์๊ฑฐ'๊ฐ ์์ต๋๋ค. ์ ๋ถ๋ฆฌ๋ฐฐ์ถ ๋ ์ฐ๋ ๊ธฐ๋ ์์์ผ๋ก์ ๊ฐ์น๋ฅผ ์ธ์ ๋ฐ์ ์ฌํ์ฉ๋์ง๋ง, ์๋ชป ๋ถ๋ฆฌ๋ฐฐ์ถ ๋๋ฉด ๊ทธ๋๋ก ํ๊ธฐ๋ฌผ๋ก ๋ถ๋ฅ๋์ด ๋งค๋ฆฝ, ์๊ฐ๋๊ธฐ ๋๋ฌธ์๋๋ค. ์ฐ๋ฆฌ๋๋ผ์ ๋ถ๋ฆฌ ์๊ฑฐ์จ์ ๊ต์ฅํ ๋์ ๊ฒ์ผ๋ก ์๋ ค์ ธ ์๊ณ , ๋ ์ต๊ทผ ์ด๋ฌํ ์ฐ๋ ๊ธฐ ๋ฌธ์ ๊ฐ ์ฃผ๋ชฉ๋ฐ์ผ๋ฉฐ ๋์ฑ ๋ง์ ์ฌ๋์ด ๋ถ๋ฆฌ์๊ฑฐ์ ๋์ฐธํ๋ ค ํ๊ณ  ์์ต๋๋ค. ํ์ง๋ง '์ด ์ฐ๋ ๊ธฐ๊ฐ ์ด๋์ ์ํ๋์ง', '์ด๋ค ๊ฒ๋ค์ ๋ถ๋ฆฌํด์ ๋ฒ๋ฆฌ๋ ๊ฒ์ด ๋ง๋์ง' ๋ฑ ์ ํํ ๋ถ๋ฆฌ์๊ฑฐ ๋ฐฉ๋ฒ์ ์๊ธฐ ์ด๋ ต๋ค๋ ๋ฌธ์ ์ ์ด ์์ต๋๋ค.

๋ฐ๋ผ์, ์ฐ๋ฆฌ๋ ์ฐ๋ ๊ธฐ๊ฐ ์ฐํ ์ฌ์ง์์ ์ฐ๋ ๊ธฐ๋ฅผ Segmentation ํ๋ ๋ชจ๋ธ์ ๋ง๋ค์ด ์ด๋ฌํ ๋ฌธ์ ์ ์ ํด๊ฒฐํด๋ณด๊ณ ์ ํฉ๋๋ค. ๋ฌธ์  ํด๊ฒฐ์ ์ํ ๋ฐ์ดํฐ์์ผ๋ก๋ ์ผ๋ฐ ์ฐ๋ ๊ธฐ, ํ๋ผ์คํฑ, ์ข์ด, ์ ๋ฆฌ ๋ฑ 11 ์ข๋ฅ์ ์ฐ๋ ๊ธฐ๊ฐ ์ฐํ ์ฌ์ง ๋ฐ์ดํฐ์์ด ์ ๊ณต๋ฉ๋๋ค.
# 
- Dataset ์ค๋ช
  - 512 x 512 ํฌ๊ธฐ์ train 2617์ฅ (80%) , public test 417์ฅ (10%) , private test 420์ฅ(10%) 
  - ์ด 11๊ฐ์ class ์กด์ฌ 
     - Background, UNKNOWN, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
  - coco format์ผ๋ก images , annotations ์ ๋ณด ์กด์ฌ
    - images : id, height , width, filename
    - annotatins : id, segmentation mask , bbox, area, category_id , image_id
    
- ํ๊ฐ๋ฐฉ๋ฒ 
    - Semantic Segmentation : mAP50 
    - Object Detection : mIoU

<br></br>
## ๐ ๋ฌธ์  ์ ์ ๋ฐ ํด๊ฒฐ ๋ฐฉ๋ฒ <a name = 'Solution'></a>
- ํด๋น ๋ํ์ ๋ํ ๋ฌธ์ ๋ฅผ ์ด๋ป๊ฒ ์ ์ํ๊ณ , ์ด๋ป๊ฒ ํ์ด๊ฐ๋์ง, ์ต์ข์ ์ผ๋ก๋ ์ด๋ค ์๋ฃจ์์ ์ฌ์ฉํ์๋์ง์ ๋ํด์๋ ๊ฐ์์ wrap up report์์ ๊ธฐ์ ํ๊ณ  ์์ต๋๋ค. 
    - [๊น์ ์ง wrapup report](https://www.notion.so/Object-Segmentation-798ebd0a47d544bc95148cff5804a600)
    - [๊นํ์ฝ wrapup report](https://maihon.oopy.io/study/boostcamp/p-stage/segmentation-detection/wrapup-report)
    - [๊นํจ์ง wrapup report](https://vimhjk.oopy.io/f31d818e-5128-4a2e-b860-07022002cb48)
    - [๋ฐ์ฑ๋ฐฐ wrapup report](https://songbae.oopy.io/2da200fe-28cf-4c3e-ae7b-f64b312a30dc)
    - [๋ฐ์ฑํ wrapup report](https://www.notion.so/Wrap-Up-Report-Stage-3-4ff86742dfb14a4383f620b7fbe13fd1)
    - [์คํ๋ฆฐ wrapup report](https://www.notion.so/Wrap-up-Pstage3-Semantic-Segmentation-2679c48f500a40f5bf7d7ffb227b8e46)

- ์ report์๋ ๋ํ๋ฅผ ์ฐธ๊ฐํ ํ, ๊ฐ์ธ์ ํ๊ณ ๋ ํฌํจ๋์ด์์ต๋๋ค. 
- ํํ๋ก์ ํธ๋ฅผ ์งํํ๋ฉฐ ํ์ ํด๋ก ์ฌ์ฉํ๋ [Notion](https://www.notion.so/1cdc0eddd3d649b68eebd94e27dc8655?v=b17e11d3c44148bc80dddf4c24b9cabf)๋ด์ฉ๋ ํด๋น ๋งํฌ์ ์ ์ํ์๋ฉด ํ์ธํ์ค ์ ์์ต๋๋ค.

<br></br>
## ๐ป CODE ์ค๋ช<a name = 'Code'></a>
- ์์ธํ CODE ์ค๋ช์ ๊ฐ์ธ ํด๋ ๋ด README.md๋ฅผ ํตํด ํ์ธํ์ค ์ ์์ต๋๋ค. 
    - [๊น์ ์ง](./Yuji-Kim)
    - [๊นํ์ฝ](./Hongyeob-Kim)
    - [๊นํจ์ง](./Hyojin-Kim)
    - [๋ฐ์ฑ๋ฐฐ](./SongBae)
    - [๋ฐ์ฑํ](./Seonghoon-Park)
    - [์คํ๋ฆฐ](./Hyerin-Oh)