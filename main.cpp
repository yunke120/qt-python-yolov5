#include "mainwindow.h"
#include <QApplication>

// [git](https://blog.csdn.net/godaa/article/details/123880957 )

// 1920*1080 960*540 640*360

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
