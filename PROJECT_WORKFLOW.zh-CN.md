# 目标检测工程说明文档

本文档基于当前仓库内已经跑通的新能源车牌检测流程，总结两部分内容：

1. 目标检测推理工程的主要工作原理，以及涉及的关键代码。
2. 训练自定义模型时涉及的主要代码结构和数据流。

## 1. 工程整体分层

当前工程可以理解为三层结构：

1. 业务入口层：负责解析命令行参数、指定模型、指定输入源、控制输出结果。
2. Ultralytics 运行层：负责真正加载模型、执行预测或训练、组织结果对象。
3. 数据与产物层：负责数据集 YAML、标签文件、训练输出权重、推理输出图片和 JSON。

对应到当前仓库，最常用的入口文件有：

1. `examples/model_test.py`：推理入口。
2. `ccpd_green/train_ccpd_green.py`：训练入口。
3. `ccpd_green/generate_yolo_labels.py`：标签生成入口。
4. `ccpd_green/ccpd_green.yaml`：训练/验证/测试数据集配置。

## 2. 目标检测推理工程主要工作原理

### 2.1 推理入口做了什么

推理入口在 `examples/model_test.py`。

它主要完成以下步骤：

1. 解析命令行参数，如模型路径、输入源、置信度、IoU、设备、输出目录等。
2. 将仓库根目录加入 `sys.path`，确保可以直接导入本地 `ultralytics` 包。
3. 加载模型：`model = YOLO(args.model)`。
4. 如果用户指定了 `--target`，则把类别名或类别 ID 转成 Ultralytics 可接受的 `classes` 参数。
5. 调用 `model.predict(...)` 执行检测。
6. 遍历预测结果，打印摘要，并在需要时把每张图的检测结果保存为 JSON。

### 2.2 推理时的关键代码职责

#### 1) 参数解析

`parse_args()` 负责定义推理时可控的所有输入参数，包括：

1. `--model`：权重文件路径。
2. `--source`：单图、目录、视频、流地址等输入源。
3. `--target`：可选，只保留某一个类别。
4. `--device`：推理设备，如 `cpu` 或 `0`。
5. `--imgsz`、`--conf`、`--iou`、`--max-det`：推理阈值与尺寸控制。
6. `--save`、`--project`、`--name`：控制可视化输出目录。

#### 2) 类别过滤

`resolve_target_class()` 的作用是把业务层输入转成底层需要的类别编号列表。

例如：

1. 用户传 `--target 5`，脚本会直接把它转换为 `[5]`。
2. 用户传 `--target bus`，脚本会在 `model.names` 中查找对应类别，最后转换为 `[class_id]`。

这个函数的价值在于：业务侧可以用“类别名”操作，而底层推理最终仍然使用稳定的类别 ID。

#### 3) 真正发起推理

`main()` 中最关键的一句是：

```python
results = model.predict(...)
```

这一步把业务入口层和 Ultralytics 运行层连接起来。

### 2.3 `YOLO.predict()` 底层如何工作

`YOLO.predict()` 位于 `ultralytics/engine/model.py`。

它的核心流程是：

1. 组合默认参数和用户参数。
2. 根据任务类型加载合适的 `predictor`。
3. 调用 `predictor.setup_model(...)` 完成模型装载。
4. 执行 `predictor(...)`，返回 `Results` 对象列表。

也就是说，业务层不需要自己处理：

1. 图像预处理。
2. 张量搬运到 CPU 或 GPU。
3. 前向计算。
4. NMS 后处理。
5. 结果对象封装。

这些都由 Ultralytics 的 predictor 体系统一完成。

### 2.4 推理结果如何组织与输出

推理返回的是 `Results` 对象列表，其实现位于 `ultralytics/engine/results.py`。

当前工程实际用到的几个结果接口是：

1. `result.verbose()`：生成人类可读的检测摘要，例如某张图检测到多少个目标。
2. `result.to_json()`：把检测框、类别、置信度等转换为 JSON 文本。
3. `result.save()` 或内部绘图流程：把框选后的图片保存到 `runs/...` 目录。

`examples/model_test.py` 中对应的封装有：

1. `summarize_result()`：读取 `boxes/masks/keypoints/obb/probs/speed` 并打印简要信息。
2. `save_result_info()`：调用 `result.to_json()` 把每张图的检测信息保存成 JSON 文件。

### 2.5 推理工程的数据流

一次完整推理的数据流可以概括为：

1. 输入源路径进入 `examples/model_test.py`。
2. 入口脚本把模型权重和参数交给 `YOLO.predict()`。
3. Ultralytics predictor 完成预处理、推理、后处理。
4. 返回 `Results` 对象。
5. 业务脚本对 `Results` 进行摘要输出、JSON 导出和渲染图保存。

最终产物通常包括：

1. 框选后的图片。
2. 对应的 JSON 检测结果。
3. 控制台摘要信息。

## 3. 训练自定义模型涉及哪些主要代码结构

当前自定义模型训练链路是围绕 CCPD 绿色新能源车牌数据集搭起来的，整体包含三段：

1. 数据标签准备。
2. 数据集配置。
3. 模型训练入口。

### 3.1 标签生成结构

标签生成脚本是 `ccpd_green/generate_yolo_labels.py`。

这个脚本的设计目标是：

1. 从 CCPD 文件名中解析出车牌框坐标。
2. 读取图片宽高。
3. 转成 YOLO 检测格式的 `class x_center y_center width height`。
4. 分别写入 `labels/train` 和 `labels/val`。

关键函数如下：

#### 1) `parse_bbox_from_name()`

作用：

1. 从文件名里拆出左上角和右下角坐标。
2. 验证框是否合法。

这是 CCPD 特有的数据解析逻辑，因为它的标注嵌在文件名里，而不是 XML/JSON。

#### 2) `to_yolo_bbox()`

作用：

1. 调用 `parse_bbox_from_name()` 得到像素坐标。
2. 用 Pillow 读图像尺寸。
3. 把像素框转换为归一化 YOLO 框。

#### 3) `write_split_labels()`

作用：

1. 遍历某个 split 下的所有图片。
2. 为每张图片生成对应的 `.txt` 标签。
3. 收集错误并统计写出数量。

这三层函数拆开后，数据清洗、坐标解析、标签落盘各自独立，后面如果更换数据集或修改标签逻辑会更容易维护。

### 3.2 数据集配置结构

训练数据配置文件是 `ccpd_green/ccpd_green.yaml`。

它主要提供三类信息：

1. `train`：训练集图片目录。
2. `val`：验证集图片目录。
3. `test`：测试集图片目录。
4. `names`：类别编号与类别名映射。

当前是单类别任务：

```yaml
names:
  0: new_energy_license_plate
```

这意味着训练时所有框都属于同一个检测类别。

### 3.3 训练入口结构

训练入口脚本是 `ccpd_green/train_ccpd_green.py`。

这个脚本的职责很明确：

1. 收集训练超参数。
2. 加载预训练权重。
3. 调用 Ultralytics 训练接口。

#### 1) `parse_args()`

这里集中定义了你当前项目中最关键的训练控制项：

1. `--model`：预训练权重，如 `yolo11n.pt`。
2. `--data`：数据集 YAML。
3. `--epochs`、`--imgsz`、`--batch`：训练规模。
4. `--device`、`--workers`：设备和 DataLoader 配置。
5. `--project`、`--name`：训练输出目录。
6. `--patience`：早停控制。
7. `--degrees`、`--fliplr`、`--mosaic`、`--close-mosaic`：增强策略。

其中几个值是结合车牌任务和你的 Windows 环境专门收敛出来的：

1. `workers=0`：避免 Windows 页面文件不足时多进程加载 CUDA 失败。
2. `fliplr=0.0`：车牌字符有方向语义，不适合水平翻转。
3. `degrees=0.0`：避免引入不合理旋转样本。
4. `mosaic=0.3`：保留适度增强，但避免对小目标车牌扰动过大。

#### 2) `main()`

训练入口最关键的两句是：

```python
model = YOLO(args.model)
model.train(...)
```

这里的含义是：

1. 先从预训练权重构建模型。
2. 再把数据集、训练轮数、增强策略、输出目录等配置交给 Ultralytics 训练引擎。

### 3.4 `YOLO.train()` 底层如何工作

`YOLO.train()` 位于 `ultralytics/engine/model.py`。

它的核心流程包括：

1. 合并默认配置、模型配置和用户传入参数。
2. 构建 trainer。
3. 用权重和模型结构初始化训练模型。
4. 调用 `self.trainer.train()` 进入正式训练循环。
5. 训练结束后自动加载最佳权重或最后权重，并把指标挂回模型对象。

换句话说，业务侧训练脚本只负责“把参数组织好”，真正的：

1. 数据集加载。
2. batch 构造。
3. 优化器创建。
4. 前向和反向传播。
5. 验证评估。
6. 最佳权重保存。

都是由 Ultralytics 的 trainer 体系接管。

### 3.5 训练产物结构

一次完整训练后，通常会在 `runs/train/<name>/` 下生成：

1. `weights/best.pt`：最佳验证结果对应的权重。
2. `weights/last.pt`：最后一个 epoch 的权重。
3. `args.yaml`：本次训练的参数快照。
4. `results.csv`：每个 epoch 的指标记录。
5. 各类标签图、训练 batch 可视化图。

当前你已经实际使用过的是：

1. `runs/train/ccpd-green-yolo11n-8gb4/weights/best.pt`：作为新能源车牌模型的推理权重。

## 4. 推理与训练两条链路如何衔接

整个工程的闭环是这样的：

1. 用 `generate_yolo_labels.py` 从原始图片文件名生成 YOLO 标签。
2. 用 `ccpd_green.yaml` 描述数据集结构和类别。
3. 用 `train_ccpd_green.py` 调 `YOLO.train()` 完成微调训练。
4. 从 `runs/train/.../weights/best.pt` 取出最佳模型。
5. 用 `examples/model_test.py` 调 `YOLO.predict()` 在新图片上做检测。
6. 把推理结果保存为框选图和 JSON。

也就是说：

1. 训练阶段产出的 `best.pt` 是推理阶段的输入。
2. 数据标注脚本决定了训练样本质量。
3. 数据集 YAML 决定了训练时如何找到图片和标签。
4. 入口脚本决定了业务层如何使用底层 Ultralytics 能力。

## 5. 当前工程里最关键的文件清单

如果只保留一组最核心文件，可以优先看这些：

1. `examples/model_test.py`：推理入口与结果导出。
2. `ccpd_green/train_ccpd_green.py`：训练入口与超参数控制。
3. `ccpd_green/generate_yolo_labels.py`：自定义标签生成逻辑。
4. `ccpd_green/ccpd_green.yaml`：训练数据集配置。
5. `ultralytics/engine/model.py`：`YOLO.predict()` 与 `YOLO.train()` 的底层分发入口。
6. `ultralytics/engine/results.py`：推理结果对象与 JSON/文本输出能力。

## 6. 总结

这个目标检测工程的本质不是“从零手写一个检测框架”，而是：

1. 以 Ultralytics YOLO11 作为底层通用引擎。
2. 在业务层补齐自己的推理入口、数据标签转换、训练入口和运行环境脚本。
3. 用标准化的 `YOLO.predict()` 和 `YOLO.train()` 接口，把业务问题映射到底层检测框架能力上。

因此，如果后续还要继续扩展，优先考虑的通常不是改底层引擎，而是扩展业务层：

1. 增加更稳的结果汇总逻辑。
2. 增加批量评估脚本。
3. 增加更多数据清洗和标签校验逻辑。
4. 根据任务需要再调整训练超参数和推理阈值。