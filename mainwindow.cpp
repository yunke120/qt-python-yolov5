#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QThread>



extern QQueue<cv::Mat> videoFrameQueue;
extern QMutex videoMutex;

extern QQueue<ROI_FRAME> roiFrameQueue;
extern QMutex detectMutex;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    setWindowTitle("Qt_Python_Yolov5检测Demo");

    qRegisterMetaType<Mat>("Mat"); /* 在Qt元对象中注册Mat类型 */
    qRegisterMetaType<ROI_FRAME>("ROI_FRAME"); /* 在Qt元对象中注册ROI_FRAME类型 */

    slotsInit();
    playerInit();

}

MainWindow::~MainWindow()
{
    playerDeinit();
    delete ui;
}

void MainWindow::slotsInit()
{
    connect(ui->btnOpenVideo,  &QPushButton::clicked, this, &MainWindow::slotBtnOpenVideo);
}

void MainWindow::playerInit()
{
#ifdef USING_DETECT
    /* python 算法初始化 */
    pDetecter = new DetectImage();
    pDetecter->setRun(true);
    pDetecter->start();
#endif
//    pPlayer = new VideoPlayer("rtsp://192.168.144.25:8554/main.264"); /* 打开rtsp流 rtsp://192.168.2.119/554*/
//    pPlayer = new VideoPlayer("rtsp://192.168.144.119/554"); /* 打开rtsp流 rtsp://192.168.2.119/554*/
    pPlayer = new VideoPlayer(0); /* 打开usb摄像头 */
    pVideoTimer = new QTimer(this);
    connect(pVideoTimer, &QTimer::timeout, this, &MainWindow::slotVideoTimerOut);
}

void MainWindow::playerDeinit()
{
    if(ui->btnOpenVideo->text() == "关闭视频")
        slotBtnOpenVideo();
#ifdef USING_DETECT
    pDetecter->setRun(false);
    pDetecter->quit();
    pDetecter->wait();
    delete pDetecter;
    pDetecter = nullptr;
#endif
    pPlayer->quit();
    pPlayer->wait();
    delete pPlayer;
    pPlayer = nullptr;

}

void MainWindow::startPlayer()
{
#ifdef USING_DETECT
    pDetecter->setPause(true);
//    pDetecter->start();
#endif
    pPlayer->setRun(true);
    pPlayer->start();
    pVideoTimer->start(1);
}

void MainWindow::stopPlayer()
{
    pPlayer->setRun(false);
#ifdef USING_DETECT
    pDetecter->setPause(false);
#endif
    pVideoTimer->stop();
}


QImage MainWindow::cvMat2QImage(const Mat &mat)
{
    if (mat.type() == CV_8UC1)                                      // 8-bits unsigned, NO. OF CHANNELS = 1
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        image.setColorCount(256);                                  // Set the color table (used to translate colour indexes to qRgb values)
        for (int i = 0; i < 256; i++)
            image.setColor(i, qRgb(i, i, i));

        uchar *pSrc = mat.data;                                    // Copy input Mat
        for (int row = 0; row < mat.rows; row++)
        {
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, pSrc, static_cast<size_t>(mat.cols));
            pSrc += mat.step;
        }
        return image;
    }
    else if (mat.type() == CV_8UC3)               // 8-bits unsigned, NO. OF CHANNELS = 3
    {
        const uchar *pSrc = reinterpret_cast<const uchar*>(mat.data);  // Copy input Mat
        QImage image(pSrc, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_RGB888); // Create QImage with same dimensions as input Mat
        return image.rgbSwapped();
    }
    else if (mat.type() == CV_8UC4)
    {
        const uchar *pSrc = reinterpret_cast<const uchar*>(mat.data);
        QImage image(pSrc, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_ARGB32);
        return image.copy();
    }
    else
    {
        return QImage();  // Mat could not be converted to QImage
    }
}

void MainWindow::slotVideoTimerOut()
{
    pVideoTimer->stop();

#ifdef USING_DETECT
    detectMutex.lock();
    bool isok = roiFrameQueue.isEmpty();
    detectMutex.unlock();
#else
    videoMutex.lock();
    bool isok = videoFrameQueue.isEmpty();
    videoMutex.unlock();
#endif
    if(!isok)
    {
#ifdef USING_DETECT
        detectMutex.lock();
        ROI_FRAME srcFrame = roiFrameQueue.dequeue();
//        int size = roiFrameQueue.size();
        detectMutex.unlock();
//        qDebug() << "roi:" << size;
        QImage img = cvMat2QImage(srcFrame.frame);
        int size = srcFrame.classList.size();
        if(size > 0)
        {
            ui->textEdit->append("----------------------------------------");
            for (int i = 0; i < srcFrame.classList.size(); i++) {
                QString _class = srcFrame.classList.value(i);
                QString _conf = srcFrame.confList.value(i);
                QString text = _class + ": " + _conf;
                ui->textEdit->append(text);
            }
        }


#else
        videoMutex.lock();
        Mat srcFrame = videoFrameQueue.dequeue();
        videoMutex.unlock();
        QImage img = cvMat2QImage(srcFrame);
#endif

        ui->labelVideo->setImage(img); /* 显示视频，使用QPainter绘制 */
    }

    pVideoTimer->start(5);
}


void MainWindow::slotBtnOpenVideo()
{
    if(ui->btnOpenVideo->text() == "打开视频")
    {
        this->startPlayer();
        ui->btnOpenVideo->setText("关闭视频");
    }
    else
    {
        this->stopPlayer();
        ui->btnOpenVideo->setText("打开视频");
    }
}
