
# Graph Attention Transformer Network for Multi-Label Image Classification



### Requirements
Please, install the following packages
- numpy
- pytorch (1.*) (1.4 and 1.9 are tested)
- torchnet
- torchvision
- tqdm
- networkx

### Download best checkpoints  
checkpoint/coco/model_best_89.2803.pth.tar and checkpoint/voc/model_best_96.3178.pth.tar ([BaiDu](https://pan.baidu.com/s/1iF0qLoeJUG-7PiPuUkdNmA)),(password:oq7s)


```sh
python train_gatn.py data/coco --image-size 448 --workers 8 --batch-size 16 --lr 0.03 --learning-rate-decay 0.1 --epoch_step 20 30 --embedding model/embedding/coco_glove_word2vec_80x300_ec.pkl --t1 0.2 --device_ids 0
```

python train_gatn.py data/coco --image-size 448 --workers 8 --batch-size 16 --lr 0.03 --learning-rate-decay 0.1 --epoch_step 20 30 --embedding model/embedding/coco_glove_word2vec_80x300_ec.pkl --t1 0.2 --device_ids 0 -e --resume checkpoint/coco/coco/model_best_89.2803.pth.tar

python train_gatn_voc.py data/voc --image-size 448 --workers 8 --batch-size 16 --lr 0.03 --learning-rate-decay 0.1 --epoch_step 20 30 --embedding model/embedding/voc_glove_word2vec.pkl --t1 0.2 --device_ids 0

python train_gatn_voc.py data/voc --image-size 448 --workers 8 --batch-size 16 --lr 0.03 --learning-rate-decay 0.1 --epoch_step 20 30 --embedding model/embedding/voc_glove_word2vec.pkl --t1 0.2 --device_ids 0 -e --resume checkpoint/voc/voc/model_best_96.3178.pth.tar



### How to cite this work?
```
@article{yuan2022graph,
  title={Graph Attention Transformer Network for Multi-Label Image Classification},
  author={Yuan, Jin and Chen, Shikai and Zhang, Yao and Shi, Zhongchao and Geng, Xin and Fan, Jianping and Rui, Yong},
  journal={arXiv preprint arXiv:2203.04049},
  year={2022}
}
```