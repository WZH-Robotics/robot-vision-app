#include "zhangVisualization.h"
#include <QImage>
#include <QPainter>
#include <cstring>
#include <cmath>

void ReprojectPoints(const ZhangVisData& data, int nImgIdx,
                     int* pu, int* pv)
{
    KMatrix mA(3,3);
    mA[0][0] = data.vX[0]; mA[1][1] = data.vX[1];
    mA[0][2] = data.vX[2]; mA[1][2] = data.vX[3];
    mA[2][2] = 1.0;
    double dK1 = data.vX[4], dK2 = data.vX[5];

    KHomogeneous Pj(data.vX.Cut(6 + nImgIdx*6, 6 + nImgIdx*6 + 5));
    KMatrix mR = Pj.R();
    KVector vT = Pj.t();

    for (int i = 0; i < data.nPoints; i++) {
        KVector vM(3);
        vM[0] = data.mModelRaw[0][i];
        vM[1] = data.mModelRaw[1][i];
        vM[2] = 0.0;

        KVector vXc = mR * vM + vT;
        double xn = vXc[0] / vXc[2];
        double yn = vXc[1] / vXc[2];
        double r2 = xn*xn + yn*yn;
        double dr = 1.0 + dK1*r2 + dK2*r2*r2;

        pu[i] = (int)(mA[0][0] * xn * dr + mA[0][2] + 0.5);
        pv[i] = (int)(mA[1][1] * yn * dr + mA[1][2] + 0.5);
    }
}

void DrawCalibrationResult(KImageColor& icCanvas, const ZhangVisData& data, int nImgIdx)
{
    int nW = icCanvas.Col(), nH = icCanvas.Row();
    memset(icCanvas.Address(), 255, icCanvas.Size() * sizeof(KCOLOR32));

    int* pu = new int[data.nPoints];
    int* pv = new int[data.nPoints];
    ReprojectPoints(data, nImgIdx, pu, pv);

    // Draw onto QImage for QPainter access
    QImage qImg(nW, nH, QImage::Format_RGB32);
    qImg.fill(Qt::white);
    QPainter painter(&qImg);
    painter.setRenderHint(QPainter::Antialiasing);

    // Draw grid lines (blue)
    QPen penGrid(Qt::blue, 1);
    painter.setPen(penGrid);

    for (int r = 0; r < data.nRows; r++)
        for (int c = 0; c < data.nCols - 1; c++) {
            int i0 = r * data.nCols + c, i1 = i0 + 1;
            painter.drawLine(pu[i0], pv[i0], pu[i1], pv[i1]);
        }

    for (int c = 0; c < data.nCols; c++)
        for (int r = 0; r < data.nRows - 1; r++) {
            int i0 = r * data.nCols + c, i1 = i0 + data.nCols;
            painter.drawLine(pu[i0], pv[i0], pu[i1], pv[i1]);
        }

    // Draw reprojected points (red)
    QPen penReproj(Qt::red, 2);
    painter.setPen(penReproj);
    for (int i = 0; i < data.nPoints; i++)
        painter.drawEllipse(QPoint(pu[i], pv[i]), 3, 3);

    // Draw observed points (green)
    QPen penObs(Qt::darkGreen, 2);
    painter.setPen(penObs);
    for (int i = 0; i < data.nPoints; i++) {
        int uo = (int)(data.mImageRaw[nImgIdx][0][i] + 0.5);
        int vo = (int)(data.mImageRaw[nImgIdx][1][i] + 0.5);
        painter.drawEllipse(QPoint(uo, vo), 2, 2);
    }

    // Legend
    QFont font("Arial", 10);
    painter.setFont(font);
    painter.setPen(Qt::black);
    painter.drawText(10, 18, QString("Image%1").arg(nImgIdx + 1));
    painter.setPen(Qt::red);
    painter.drawText(10, 34, "Red: reprojected");
    painter.setPen(Qt::darkGreen);
    painter.drawText(10, 50, "Green: observed");

    painter.end();

    // Copy QImage back to KImageColor
    QImage converted = qImg.convertToFormat(QImage::Format_RGB32);
    memcpy(icCanvas.Address(), converted.bits(), nH * nW * sizeof(KCOLOR32));

    delete[] pu;
    delete[] pv;
}

void SaveCalibrationImages(const ZhangVisData& data, const QString& outputDir)
{
    const int nW = 640, nH = 480;

    for (int j = 0; j < data.nImages; j++) {
        KImageColor icCanvas(nH, nW);
        DrawCalibrationResult(icCanvas, data, j);

        QImage qImg(nW, nH, QImage::Format_RGB32);
        memcpy(qImg.bits(), icCanvas.Address(), nH * nW * sizeof(KCOLOR32));

        QString path = QString("%1/calibration_image%2.png").arg(outputDir).arg(j + 1);
        qImg.save(path);
    }
}
