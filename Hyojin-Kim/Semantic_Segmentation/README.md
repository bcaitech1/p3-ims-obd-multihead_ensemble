Baseline-v2
===
## Usage
- `cd ~`
- `mkdir baselinev2`
- `cd baselinev2`
- `git clone -b vim_hjk --single-branch https://github.com/bcaitech1/p3-ims-obd-multihead_ensemble.git`
### Train
- `python main.py --config "your config"`

### Evaluation
- `python eval.py --eval_config "your eval config"`

### Inference
- If you want to inference all the test batches,
    - `python inference.py`

- x is Number of images to use in inference.
    - `python inference.py --limit x`
---
### Demo
- ![demo](./prediction/PAN/29.jpg) 

