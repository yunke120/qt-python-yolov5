#ifndef VIDEOPLAYER_H
#define VIDEOPLAYER_H

#include <QThread>
#include <QImage>
#include <QMutex>
#include <QQueue>
#include <QDebug>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
class VideoPlayer : public QThread
{
    Q_OBJECT
public:
    VideoPlayer(QString);
    VideoPlayer(int);

    void setAddr(QString s)
    {
        ISSTR_ADDR = true;
        addr = s.toStdString();
    }

    void setAddr(int i)
    {
        ISSTR_ADDR = false;
        addr_int = i;
    }

    void setRun(bool flag)
    {
        QMutexLocker locker(&mutex);
        IS_RUN = flag;
    }



private:
    string addr;
    int addr_int;
    bool IS_RUN = false;
    QMutex mutex;
    VideoCapture *pCap = nullptr;
    bool ISSTR_ADDR = false;



signals:
    void sigSendErrorCode(int); /* 发送错误码 */

protected:
    virtual void run() Q_DECL_OVERRIDE;
};

#endif // VIDEOPLAYER_H
