#ifndef ZHANGVISUALIZATION_H
#define ZHANGVISUALIZATION_H

#include "kfc.h"
#include <QString>

struct ZhangVisData {
    KMatrix mModelRaw;       // 3xN model points
    KMatrix mImageRaw[7];    // 3xN image points per image
    KVector vX;              // optimized 48-dim state vector
    int     nImages;
    int     nPoints;
    int     nCols;           // grid columns (12)
    int     nRows;           // grid rows (13)
};

// Reproject model points for one image, returns pixel coords in pu[], pv[]
void ReprojectPoints(const ZhangVisData& data, int nImgIdx,
                     int* pu, int* pv);

// Draw projected grid + observed/reprojected points onto a color image
void DrawCalibrationResult(KImageColor& icCanvas, const ZhangVisData& data, int nImgIdx);

// Save all 7 images to output folder as PNG
void SaveCalibrationImages(const ZhangVisData& data, const QString& outputDir);

#endif // ZHANGVISUALIZATION_H
