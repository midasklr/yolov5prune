# yolov5剪枝





| model   | input | map   |
| ------- | ----- | ----- |
| yolov5s | 512   | 0.37  |
| yolov5s | 320   | 0.275 |
| yolov5s | 416   |       |
|         |       | 0.195 |









## 可以裁剪的bn层

1. conv.bn
2. cv1, cv2, cv3
3. m.0.cv1.bn
4. BottleNeck.add = False的cv2.bn

## 不能裁剪的bn层

m.0.cv2.bn，m.1.cv2.bn，m.2.cv2.bn