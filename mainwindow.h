#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QStringList>
#include <QPixmap>
#include <QDateTime>
#include <QPushButton>
#include <QDir>
#include <QMetaType>

#include "detectimage.h"
#include "videoplayer.h"
#include "paintlabel.h"


#if _MSC_VER >= 1600
#pragma execution_character_set("utf-8")
#endif

Q_DECLARE_METATYPE(ROI_FRAME);
Q_DECLARE_METATYPE(Mat);


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:

    void slotsInit(void);

    /*********************** 视频播放 ***********************/
    void playerInit(void);      /* 播放器初始化 */
    void playerDeinit(void);    /* 反初始化 */
    void startPlayer(void);     /* 开始播放 */
    void stopPlayer(void);      /* 停止播放 */

    /*********************** 其他 ***********************/
    QImage cvMat2QImage(const cv::Mat &mat);

private slots:

    void slotVideoTimerOut(void);    /* 显示视频定时器 超时中断函数 */
    void slotBtnOpenVideo(void);     /* 打开视频按钮槽函数   */

public slots:
    void slotResetThread(void);


private:
    Ui::MainWindow *ui;

    VideoPlayer *pPlayer;          /* 播放器线程类   */
    DetectImage *pDetecter;        /* 图像检测线程类 */
    QTimer *pVideoTimer;
    Mat frame;

};

#endif // MAINWINDOW_H
