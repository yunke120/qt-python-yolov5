文件夹`yolov5`为存放检测算法的文件夹，其中

1. `yolov5`默认文件未做修改，`temp.py`为在`detect.py`的基础上修改的启动文件。
2. 测试用权重文件为`yolov5s.pt`，下载地址为[链接](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt)，您可以下载其他的模型[文件](https://github.com/ultralytics/yolov5/releases)，并将其放在`yolov5`文件夹下，并修改`temp.py`的初始化变量`self.weights`即可。