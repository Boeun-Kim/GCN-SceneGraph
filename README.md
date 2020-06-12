# Graph R-CNN for Scene Graph Generation (ECCV 2018) 

이 페이지는 2020년도 제어자동화특강(GNN) 프로젝트의 일환으로 아래 논문을 리뷰하고 저자의 코드를 분석한 내용을 포함한다.

> Yang, Jianwei, et al. "Graph r-cnn for scene graph generation." *Proceedings of the European conference on computer vision (ECCV)*. 2018.



# Preparation

### Docker

코드를 실행하는데 필요한 툴킷 및 라이브러리등을 설치한 도커 이미지를 dockerhub에 업로드 하였고, 아래 명령어로 pull하여 사용 가능하다.

```
docker pull kbe36/graph-rcnn:v2
```

설치 툴킷 및 cuda 버전

- Python 3.6+
- Pytorch 1.0
- CUDA 8.0+


### Dataset

VisualGenome 데이터셋. 다음 링크를 참고하여 따라 데이터셋을 사용 가능한 형태로 변경 후 아래 경로에 저장한다.

```
datasets/vg_bm/imdb_1024.h5
datasets/vg_bm/bbox_distribution.npy
datasets/vg_bm/proposals.h5
datasets/vg_bm/VG-SGG-dicts.json
datasets/vg_bm/VG-SGG.h5
```



# Run

### Train

...

### Evaluate

...



# Analysis of the code

### ...

...

### 



# Acknowledgement

이 코드는 논문 저자의 github에 업로드된 implementation을 따른 것이며, 실행이 용이하도록 일부 수정하여 업로드한 것이다. 

> https://github.com/jwyang/graph-rcnn.pytorch

