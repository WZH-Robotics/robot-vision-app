#ifndef CORNER_DETECTION_H
#define CORNER_DETECTION_H

#include <QString>
#include <QStringList>
#include <QImage>
#include <vector>

namespace corner_detection {

struct Point {
    int x, y;
    float response;
};

struct ChessboardSpecs {
    double mmHeight;
    double mmWidth;
    int height;
    int width;
};

struct CornerPoint {
    double u, v;
};

ChessboardSpecs readChessboardSpecs(const QString& filePath);
QImage toGrayscale(const QImage& image);
bool findChessboardCorners(const QImage& grayImage, int rows, int cols,
                           std::vector<CornerPoint>& corners);
void processImages(const QString& directoryPath, const QString& outputDirectoryPath,
                   int start, int end, int rows, int cols);
QStringList runCornerDetection(const QString& dataDir);

} // namespace corner_detection

#endif // CORNER_DETECTION_H
