# QT_YoloV5_Demo

## 目录

1. [仓库更新 Top News](https://gitee.com/yunke120/substation-robot/tree/master/#top-news)
2. [所需环境 Environment](https://gitee.com/yunke120/substation-robot/tree/master/#%E6%89%80%E9%9C%80%E7%8E%AF%E5%A2%83)
3. [文件下载 Download](https://gitee.com/yunke120/substation-robot/tree/master/#%E6%96%87%E4%BB%B6%E4%B8%8B%E8%BD%BD)
4. [使用方式 How2use](https://gitee.com/yunke120/substation-robot/tree/master/#%E4%BD%BF%E7%94%A8%E6%96%B9%E5%BC%8F)
5. [参考资料 Reference](https://gitee.com/yunke120/substation-robot/tree/master/#%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99)

## Top News



## 所需环境

1. 本机环境
   - QT 5.13.0
   - Python 3.8.10
   - VS2017 x64
   - cuda 11.4
   - opencv 4.5.2
2. 所需环境
   - QT >= Qt 5
   - Python >= 3.7 (根据算法所用版本，例如`yolov5`需要`Python>=3.7.0`，`PyTorch>=1.7`)

## 文件下载

```
git clone https://gitee.com/yunke120/qt-python-yolov5.git
```

## 使用方式

### 文件结构

```
.
├── bin/               /* release 目录 */
├── detectimage.cpp    /* 检测图片线程源文件 */ 
├── detectimage.h      /* 检测图片线程头文件 */
├── inc/               /* 外部头文件 */
├── libs/              /* 外部库文件 */
├── main.cpp           /* 启动入口 */
├── mainwindow.cpp     /* 主线程源文件 */
├── mainwindow.h       /* 主线程（GUI）头文件 */
├── mainwindow.ui      /* UI文件 */
├── paintlabel.cpp     /* 重写QLabel源文件 */
├── paintlabel.h       /* 重写QLabel头文件 */
├── README.md          /* 说明 */
├── Demo.pro          /* 工程文件 */
├── videoplayer.cpp    /* 接收视频线程源文件 */
└── videoplayer.h      /* 接收视频线程头文件 */
```

```
/bin文件结构：
├── opencv_videoio_ffmpeg452_64.dll /* 所需动态库 */
├── opencv_world452.dll
├── openh264-1.8.0-win64.dll
├── python38.dll
├── python3.dll
├── Demo.exe                       
└── yolov5/                         /* 你的算法 */
```

​		在运行此程序之前，需要完成以下几步：

​		1. 请确保您的`Python`算法能够单独运行，这是为了确保能够成功调用`Python`算法。

​		以`YoloV5`为例，打开`yolov5`文件夹，运行

```
python detect.py --weights yolov5s.pt --source data/images/bus.jpg
```

​		运行成功即可。

![1658030901940](figures/1658030901940.png)

  2. 修改目录`bin`、`libs`、`inc`下的`python`相关文件。这些文件需要与你在步骤1中使用的`Python`环境对应，查看当前`Python`安装目录，执行命令`where python`。

     ![1658032061632](figures/1658032061632.png)