# 显式设备接入与真实MPS训练（2026-11-16）

已增加 `module3_instances.py --device cpu/mps`，默认CPU。模型、输入图像和训练targets均显式放到所选设备；评测/保存前将预测转回CPU。MPS不可用会明确失败，不做静默CPU回退。checkpoint增加MPS随机状态，加载时先映射CPU再按目标设备恢复。

## 真实四图对照

同一四图、官方初始化权重与种子42，输入256/384、冻结backbone、100步训练。使用独立 `module3_device_check.py`，MPS在沙箱外执行；未启用PYTORCH_ENABLE_MPS_FALLBACK。两组checkpoint重载预测一致，19项回归测试通过。

|后端|训练100步|四图AP50|首个训练后预测耗时|
|---|---:|---:|---:|
|CPU|21.59秒|62.62%|0.091秒|
|MPS|150.94秒|37.62%|0.591秒|

MPS模型参数确实位于mps:0，真实训练前向/反向、SGD更新和预测已跑通。每步更新后采样的最大live张量内存约350.85 MiB、driver内存约1234.73 MiB；不是整个进程或训练瞬间峰值，不能外推为正式训练显存需求。

MPS本轮约为CPU的7倍耗时。每个后端只跑一次，开始阶段存在短暂并行，且计时包括首轮编译/传输开销，因此不能泛化为Apple GPU永远更慢。CPU与MPS的随机算子/浮点数值路径不同，相同种子不保证学习轨迹一致；AP差异尚未逐项定位。

## 结论

本机GPU可用的问题已解决，但当前四图门槛仍未通过；不启动扩容或test推理。继续四图排查时默认CPU更合适。下一轮应保持数据与固定预算，集中诊断RPN召回和ROI背景偏置，尝试更合适的分辨率/候选配置；不能仅靠换MPS或租服务器期待模型自动达标。更大batch、分辨率或全量数据是否需要CUDA服务器，应另做吞吐与容量实测。

原始产物：artifacts/module3_device_cpu_v1、artifacts/module3_device_mps_v1。两组保留protocol、checkpoint、100步损失/梯度、预测NPZ、可视化、时间和内存采样。test/validation未用于本轮训练诊断，未同步GitHub。
