# Module 3 开发进度同步

本次同步代码、配置、测试、数据发布manifest/质量审计、全部模块三实验报告与小型预测产物。官方train/validation/test划分不变。

当前进度：MPS设备选项已实现，19项测试通过；真实四图100步对照CPU AP50=62.62%、MPS AP50=37.62%，均未通过80%诊断门槛。没有启动全量训练或正式test推理。下一步继续排查RPN召回与ROI背景偏置。

大型checkpoint和完整数据split仍留在本地，精确路径、大小与SHA256见module3_upload_inventory.json。上传的历史manifest可能引用这些未包含文件；完整实验重放需本地数据和checkpoint，并非只clone仓库即可复原全部大文件。

入口与运行记录详见module3_segmentation.md。没有上传虚拟环境、缓存、密钥或无关模块开发文件。
