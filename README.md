# STFPM 单类别实验记录：MVTec AD carpet

## 目标与数据

- 目标：先跑通工业图像异常检测与定位基线 STFPM 的一个类别。
- 数据：MVTec AD 原版 `carpet`。正常训练图 280 张；测试图中正常图 28 张、缺陷图 89 张，89 张缺陷图均有对应掩码。
- 训练只使用正常图；缺陷图及掩码用于测试评估。

## 方法与运行设置

- 代码：论文作者仓库 `gdwang08/STFPM` 的 `main.py`、`evaluate.py`。
- 教师：ImageNet 预训练 ResNet-18，固定参数；学生：同结构 ResNet-18，随机初始化。
- 特征：ResNet-18 的三个层级；输入缩放为 256 × 256。
- 训练：`carpet`，100 轮，批量 32；训练集按作者代码以 `random_state=0` 划出 20% 作验证，保存验证误差最低的学生模型。
- 正式模型：`STFPM-main/snapshots/carpet/best.pth.tar`。
- 兼容性修改：Windows 中文路径下，`cv2.imread` 无法读取缺陷掩码；`main.py` 改用 `np.fromfile` 与 `cv2.imdecode`，其余掩码处理流程不变。

## 正式测试结果

| 指标 | 本次结果 | 作者仓库 README 的 carpet 参考值 |
| --- | ---: | ---: |
| Pixel-AUC | 0.990094 | 0.990292 |
| Image-AUC | 0.962681 | 0.964286 |
| PRO | 0.964143 | 0.966061 |

正式异常热图：`STFPM-main/results/carpet_color_000_heatmap_formal.png`。该图从左到右为原图、异常热图和叠加图；高响应区域与 `ground_truth/color/000_mask.png` 的缺陷位置一致。热图颜色按单张图缩放，仅用于展示，评估使用原始异常分数。

## 范围与下一步

本次只复现了 `carpet` 单类别，尚未计算论文的 15 类平均结果。论文正文采用 100 轮，作者仓库 README 的示例命令采用 200 轮；本次按论文正文设置。下一步可保存一张正常测试图的热图，检查误报，再决定是否扩展到其他类别。
