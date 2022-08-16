#include "videoplayer.h"

QQueue<cv::Mat> videoFrameQueue;
QMutex videoMutex;

VideoPlayer::VideoPlayer(QString s)
{
    ISSTR_ADDR = true;
    addr = s.toStdString();
}

VideoPlayer::VideoPlayer(int i)
{
    ISSTR_ADDR = false;
    addr_int = i;
}

void VideoPlayer::run()
{

        if(ISSTR_ADDR)
            pCap = new VideoCapture(addr);  /* 实例化视频捕获设备 */
        else
            pCap = new VideoCapture(addr_int);
        if(!pCap->isOpened())
        {
            return;
        }
        Mat frame;
        Mat rgb_frame;
        int waitTime = 5;
        do                              /* 连接前的延时，对于拉流视频不可少 */
        {
            *pCap >> frame;
            if(!frame.empty()) break;
            msleep(10);
        }while(--waitTime);

        waitTime = 3; /* 5s内接收不到视频就退出线程 */
        while (IS_RUN) {               /* 死循环 */
            *pCap >> frame;            /* 如果frame为空 ，这个地方opencv应该做了尝试读取，时间大概是1至2秒，这里只尝试3次*/
            if(frame.empty())
            {
                waitTime --;
                qDebug() << waitTime;
                if(waitTime == 0)
                    break;
                continue;
            }
            else
            {
                waitTime = 3;

                videoMutex.lock();
                videoFrameQueue.enqueue(frame);
                videoMutex.unlock();
            }

            msleep(10);
        }

        if(waitTime == 0)
        {
            emit sigResetThread();
        }
        IS_RUN = false;
        pCap->release();
        delete pCap;
        pCap = nullptr;
        return;
}
