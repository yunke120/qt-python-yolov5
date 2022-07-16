#ifndef DETECTIMAGE_H
#define DETECTIMAGE_H

#include <QThread>
#include <QImage>
#include <QMutex>
#include <QQueue>
#include <QDebug>
#include <QTimer>

#include <python/Python.h>
#include <numpy/ndarrayobject.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>



typedef struct {
    cv::Mat frame;      /* 带有检测框的图像 */
    QStringList classList;
    QStringList confList;
}ROI_FRAME;

using namespace cv;
using namespace std;
class DetectImage : public QThread
{
    Q_OBJECT
public:
    DetectImage();

    int pythonInit(const char *module_name, const char *class_name);                      /* python算法初始化 */
    void pythonDeinit(void);                                                              /* python反初始化 */
    int loadAlgorithmModel(const char *func_name);                                        /* 加载算法模型 */
    void setPause(bool flag);
    void setRun(bool);

private:
    QImage cvMat2QImage(const Mat &mat);                                                  /* 将Mat转换为QImage类型 */
    int detectImageEx3(const char *fun, Mat srcImg, ROI_FRAME &roiFrame);

protected:
    virtual void run() Q_DECL_OVERRIDE;

private:

    PyObject *pDetect = nullptr;   /* python检测图像类                             */
    int isExist = 0;               /* python detect的返回值，判断是否检测到目标     */
    int x1,x2,y1,y2;               /* python detect的返回值，目标在原图中的矩形坐标 */
    QMutex mutex;
    bool IS_Pause = false;
    bool AlwaysRun = false;
    Mat frame;
};

#endif // DETECTIMAGE_H
