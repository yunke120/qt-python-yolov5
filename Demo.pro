#-------------------------------------------------
#
# Project created by QtCreator 2022-06-19T15:26:56
#
#-------------------------------------------------

QT       += core gui

DEFINES += USING_DETECT

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets


TARGET = Demo
TEMPLATE = app
DESTDIR     =  $$PWD/bin
OBJECTS_DIR =  $$PWD/LOGS/temp/obj
MOC_DIR     =  $$PWD/LOGS/temp/moc
UI_DIR      =  $$PWD/LOGS/temp/ui
RCC_DIR     =  $$PWD/LOGS/temp/rcc

INCLUDEPATH += $$PWD/inc
INCLUDEPATH += $$PWD/inc/python
LIBS += -L$$PWD/libs             \
              -llibopencv_world452 \
              -lpython3         \
              -lpython38

DEFINES += QT_DEPRECATED_WARNINGS

CONFIG += c++11

SOURCES += \
        detectimage.cpp \
        main.cpp \
        mainwindow.cpp \
        paintlabel.cpp \
        videoplayer.cpp

HEADERS += \
        detectimage.h \
        mainwindow.h \
        paintlabel.h \
        videoplayer.h

FORMS += \
        mainwindow.ui

RESOURCES +=
