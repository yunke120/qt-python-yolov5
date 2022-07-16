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

        while (IS_RUN) {               /* 死循环 */
            *pCap >> frame;
            if(frame.empty())
            {
//                break;
            }

            videoMutex.lock();
            videoFrameQueue.enqueue(frame);
            videoMutex.unlock();
            msleep(10);
        }
        IS_RUN = false;
        pCap->release();
        delete pCap;
        pCap = nullptr;
        return;
}
