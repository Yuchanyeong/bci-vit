global average pooling 

### 문제점 
1. 데이터셋이 너~무 적다. 
-  1-1. 그리고 그 데이터 셋이 pretrained된 domain과 달라. 
2. vit의 inducted bias? 

PYTHONPATH=. python src/train.py --config configs/default.yaml



 scp -r ycy@117.16.244.176:workspace/vit/experiments/wave_new/result /Users/yuchan-yeong


 
### 전처리의 문제? 

#### T데이터와 E데이터의 전처리를 어떻게 해야할지

#### 그리고 그걸 어떻게(어떤 데이터를 가져다가) 학습과 테스트에 적용해야할지... 

#### 5/3 데이터 증강 한번 해보자 

1. wavelet 시각화 데이터 뽑기. 
2. Loso sub마다 데이터 뽑기. 
