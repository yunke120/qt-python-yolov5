#include "paintlabel.h"

PaintLabel::PaintLabel(QGroupBox *)
{

}

void PaintLabel::paintEvent(QPaintEvent *)
{
    QPainter painter;
    painter.begin(this);
    painter.setBrush(Qt::white);
    painter.drawRect(0,0,this->width(), this->height());
    if(mImage.isNull()) return;
    QImage img = mImage.scaled(this->size(), Qt::KeepAspectRatio);
    int x = this->width() - img.width();
    int y = this->height() - img.height();
    x = x >> 2;
    y = y >> 2;
    painter.drawImage(QPoint(x, y), img);
    painter.end();
}


