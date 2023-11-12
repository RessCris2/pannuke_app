
# CoNSeP

- 解压文件夹
- 生成 coco format
- 生成 seg_mask
- 生成 seg_dist_mask
- 生成 yolo format

```python
!unzip consep.zip

!unzip MoNuSAC_images_and_annotations.zip -d train
!mv ./train/MoNuSAC_images_and_annotations/* ./train/
!unzip MoNuSAC\ Testing\ Data\ and\ Annotations.zip -d test
!mv test/MoNuSAC\ Testing\ Data\ and\ Annotations/* test/


!unzip fold_1.zip -d fold1
!unzip fold_2.zip -d fold2
!unzip fold_3.zip -d fold3

# 处理完 pannuke2mask 后，合并 train_val
rsync -av --ignore-existing val/ train/
```