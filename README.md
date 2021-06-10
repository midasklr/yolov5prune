
# yolov5模型剪枝



基于yolov5 v5.0分支进行剪枝，采用yolov5s模型。

相关原理：

Learning Efficient Convolutional Networks Through Network Slimming（https://arxiv.org/abs/1708.06519）

Pruning Filters for Efficient ConvNets（https://arxiv.org/abs/1608.08710）

相关原理见https://blog.csdn.net/IEEE_FELLOW/article/details/117236025

这里实验了三种剪枝方式

## 剪枝方法1

基于BN层系数gamma剪枝。

在一个卷积-BN-激活模块中，BN层可以实现通道的缩放。如下：

<p align="center">
<img src="img/Screenshot from 2021-05-25 00-26-23.png">
</p>

BN层的具体操作有两部分：

<p align="center">
<img src="img/Screenshot from 2021-05-25 00-28-15.png">
</p>

在归一化后会进行线性变换，那么当系数gamma很小时候，对应的激活（Zout）会相应很小。这些响应很小的输出可以裁剪掉，这样就实现了bn层的通道剪枝。

通过在loss函数中添加gamma的L1正则约束，可以实现gamma的稀疏化。

<p align="center">
<img src="img/Screenshot from 2021-05-25 00-28-52.png">
</p>



上面损失函数L右边第一项是原始的损失函数，第二项是约束，其中g(s) = |s|，λ是正则系数，根据数据集调整

实际训练的时候，就是在优化L最小，依据梯度下降算法：

​														𝐿′=∑𝑙′+𝜆∑𝑔′(𝛾)=∑𝑙′+𝜆∑|𝛾|′=∑𝑙′+𝜆∑𝛾∗𝑠𝑖𝑔𝑛(𝛾)

所以只需要在BP传播时候，在BN层权重乘以权重的符号函数输出和系数即可，对应添加如下代码:

```python
            # Backward
            loss.backward()
            # scaler.scale(loss).backward()
            # # ============================= sparsity training ========================== #
            srtmp = opt.sr*(1 - 0.9*epoch/epochs)
            if opt.st:
                ignore_bn_list = []
                for k, m in model.named_modules():
                    if isinstance(m, Bottleneck):
                        if m.add:
                            print("miss : ", k)
                            ignore_bn_list.append(i.rsplit(".", 2)[0] + ".cv1.bn")
                            ignore_bn_list.append(k + '.cv1.bn')
                            ignore_bn_list.append(k + '.cv2.bn')
                    if isinstance(m, nn.BatchNorm2d) and (k not in ignore_bn_list):
                        m.weight.grad.data.add_(srtmp * torch.sign(m.weight.data))  # L1
                        m.bias.grad.data.add_(opt.sr*10 * torch.sign(m.bias.data))  # L1
            # # ============================= sparsity training ========================== #

            optimizer.step()
                # scaler.step(optimizer)  # optimizer.step
                # scaler.update()
            optimizer.zero_grad()
```

这里并未对所有BN层gamma进行约束，详情见yolov5s每个模块 https://blog.csdn.net/IEEE_FELLOW/article/details/117536808
分析，这里对C3结构中的Bottleneck结构中有shortcut的层不进行剪枝，主要是为了保持tensor维度可以加：

<p align="center">
<img src="img/Screenshot from 2021-05-27 22-20-33.png">
</p>

实际上，在yolov5中，只有backbone中的Bottleneck是有shortcut的，Head中全部没有shortcut.

如果不加L1正则约束，训练结束后的BN层gamma分布近似正太分布：

<p align="center">
<img src="img/Screenshot from 2021-05-23 20-19-08.png">
</p>

是无法进行剪枝的。

稀疏训练后的分布：

<p align="center">
<img src="img/Screenshot from 2021-05-23 20-19-30.png">
</p>

可以看到，随着训练epoch进行，越来越多的gamma逼近0.

训练完成后可以进行剪枝，一个基本的原则是阈值不能大于任何通道bn的最大gamma。然后根据设定的裁剪比例剪枝。

剪掉一个BN层，需要将对应上一层的卷积核裁剪掉，同时将下一层卷积核对应的通道减掉。

这里在某个数据集上实验。

首先使用train.py进行正常训练：

```
python train.py --weights yolov5s.pt --adam --epochs 100
```

然后稀疏训练：

```
python train_sparsity.py --st --sr 0.0001 --weights yolov5s.pt --adam --epochs 100
```

sr的选择需要根据数据集调整，可以通过观察tensorboard的map，gamma变化直方图等选择。

训练完成后进行剪枝：

```
python prune.py --weights runs/train/exp1/weights/last.pt --percent 0.5
```

裁剪比例percent根据效果调整，可以从小到大试。裁剪完成会保存对应的模型pruned_model.pt。

微调：

```
python finetune_pruned.py --weights pruned_model.pt --adam --epochs 100
```



| model                 | sparity | map   | mode size |
| --------------------- | ------- | ----- | --------- |
| yolov5s               | 0       | 0.322 | 28.7 M    |
| sparity train yolov5s | 0.001   | 0.325 | 28.7 M    |
| 65% pruned yolov5s    | 0.001   | 0.318 | 6.8 M     |
| fine-tune             | 0       | 0.325 | 6.8 M     |

## 剪枝方法2

对于Bottleneck结构：

<p align="center">
<img src="img/Screenshot from 2021-06-05 00-06-27.png">
</p>

如果有右边的参差很小，那么就只剩下左边shortcut连接，相当于整个模块都裁剪掉。可以进行约束让参差逼近0.见train_sparsity2.py。

backbone一共有3个bottleneck，裁剪全部bottleneck：

| model                       | sparity | map   | model size |
| --------------------------- | ------- | ----- | ---------- |
| yolov5s-prune all bottlenet | 0.001   | 0.167 | 28.7 M     |
| 85%+Bottlenet               |         | 0.151 | 1.1 M      |
| finetune                    |         | 0.148 |            |

| 裁剪Bottleneck数  | map   |
| ----------------- | ----- |
| 所有bottle res    | 0.167 |
| 第2,3的bottle res | 0.174 |
| 第3的bottle res   | 0.198 |

可以看到实际效果并不好，从bn层分布也可以看到，浅层特征很少被裁减掉。

## 剪枝方法3

卷积核剪枝，那些权重很小的卷积核对应输出也较小，那么对kernel进行约束，是可以对卷积核进行裁剪的。

裁剪卷积核需要将下一层BN层对应裁剪，同时裁剪下一层卷积层的输出通道。见train_sparsity3.py

|                  | s    | model size | map   |
| ---------------- | ---- | ---------- | ----- |
| sparity train    | 1e-5 | 28.7 M     | 0.335 |
| 50% kernel prune |      | 8.4 M      | 0.151 |
| finetune         |      | 8.4 M      | 0.332 |

## 剪枝方法4

混合1和3，见train_sparsity4.py

|                             | map   | model size |
| --------------------------- | ----- | ---------- |
| conv+bn sparity train       | 0.284 | 28.7 M     |
| 85% bn prune                | 0.284 | 3.7 M      |
| 78% conv prune              | 0.284 | 3.9 M      |
| 85% bn prune+78% conv prune | 0.284 | 3.7 M      |
