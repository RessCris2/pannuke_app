
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


```