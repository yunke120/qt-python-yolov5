#ifndef PAINTLABEL_H
#define PAINTLABEL_H

#include <QObject>
#include <QLabel>
#include <QPainter>
#include <QPaintEvent>
#include <QGroupBox>

class PaintLabel : public QLabel
{
public:
    PaintLabel(QGroupBox *parent = nullptr);

    void setImage(QImage img)
    {
        mImage = img;
        update();
    }

protected:
    void paintEvent(QPaintEvent *);

private:
    QImage mImage;
};

#endif // PAINTLABEL_H
