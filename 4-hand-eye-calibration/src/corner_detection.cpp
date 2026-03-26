#include "corner_detection.h"

#include <QFile>
#include <QTextStream>
#include <QDir>
#include <QDebug>
#include <QPainter>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;

namespace corner_detection {

ChessboardSpecs readChessboardSpecs(const QString& filePath) {
    QFile file(filePath);
    ChessboardSpecs specs{0.0, 0.0, 0, 0};
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "Cannot open file for reading:" << filePath;
        return specs;
    }
    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList parts = line.split(' ');
        if (parts.size() == 2) {
            if (parts[0] == "chess_mm_height") specs.mmHeight = parts[1].toDouble();
            else if (parts[0] == "chess_mm_width") specs.mmWidth = parts[1].toDouble();
            else if (parts[0] == "chess_height") specs.height = parts[1].toInt();
            else if (parts[0] == "chess_width") specs.width = parts[1].toInt();
        }
    }
    file.close();
    return specs;
}

QImage toGrayscale(const QImage& image) {
    return image.convertToFormat(QImage::Format_Grayscale8);
}

bool findChessboardCorners(const QImage& grayImage, int rows, int cols,
                           vector<CornerPoint>& corners) {
    // QImage (Grayscale8) -> cv::Mat (no copy, shared data)
    cv::Mat gray(grayImage.height(), grayImage.width(), CV_8UC1,
                 const_cast<uchar*>(grayImage.bits()), grayImage.bytesPerLine());

    cv::Size patternSize(cols, rows);  // OpenCV: (width, height) = (cols, rows)
    vector<cv::Point2f> cvCorners;

    bool found = cv::findChessboardCorners(gray, patternSize, cvCorners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
    if (!found) return false;

    // Sub-pixel refinement
    cv::cornerSubPix(gray, cvCorners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

    corners.clear();
    corners.reserve(cvCorners.size());
    for (const auto& p : cvCorners)
        corners.push_back({(double)p.x, (double)p.y});
    return true;
}

void processImages(const QString& directoryPath, const QString& outputDirectoryPath,
                   int start, int end, int rows, int cols) {
    QDir directory(directoryPath);
    QStringList filters;
    for (int i = start; i <= end; ++i)
        filters << QString("image%1.tiff").arg(i);

    QStringList imageFiles = directory.entryList(filters, QDir::Files);
    QDir outputDir(outputDirectoryPath);
    if (!outputDir.exists()) outputDir.mkpath(".");

    foreach (const QString &imageFile, imageFiles) {
        QImage qImage(directory.filePath(imageFile));
        if (qImage.isNull()) {
            qDebug() << "Failed to load the image:" << imageFile;
            continue;
        }

        QImage grayQImage = toGrayscale(qImage);
        vector<CornerPoint> corners;
        if (!findChessboardCorners(grayQImage, rows, cols, corners)) {
            qDebug() << "Failed to find chessboard corners in" << imageFile;
            continue;
        }

        QFile outputFile(outputDir.filePath(imageFile + ".txt"));
        if (!outputFile.open(QFile::WriteOnly | QFile::Text)) {
            qDebug() << "Failed to open file for output:" << outputFile.fileName();
            continue;
        }

        QTextStream out(&outputFile);
        for (const auto& corner : corners)
            out << corner.u << " " << corner.v << "\n";
        outputFile.close();
        qDebug() << "Processed and saved corners for" << imageFile;
    }
}

QStringList runCornerDetection(const QString& dataDir) {
    QStringList msgs;

    QString calibFile = dataDir + "/calibration_object.txt";
    ChessboardSpecs specs = readChessboardSpecs(calibFile);
    msgs << QString("Loaded chessboard specs: %1x%2, square %3x%4 mm")
                .arg(specs.height).arg(specs.width)
                .arg(specs.mmHeight).arg(specs.mmWidth);

    if (specs.height == 0 || specs.width == 0) {
        msgs << "ERROR: Failed to load chessboard specs.";
        return msgs;
    }

    msgs << "Processing hand-eye images (0-14)...";
    processImages(dataDir + "/images/camera0", "./output/corners_handeye",
                  0, 14, specs.height, specs.width);
    msgs << "Hand-eye corner files saved to ./output/corners_handeye/";

    msgs << "Processing internal calibration images (0-41)...";
    processImages(dataDir + "/internal_images/camera0",
                  "./output/corners_internal", 0, 41, specs.height, specs.width);
    msgs << "Internal corner files saved to ./output/corners_internal/";

    // Visualization — generate for ALL successfully detected images
    QString vizDir = "./output/1.corner_detection";
    QDir().mkpath(vizDir);

    auto visualize = [&](const QString& imgDir, const QString& cornerDir,
                         int start, int end, const QString& prefix) {
        int vizCount = 0;
        for (int idx = start; idx <= end; idx++) {
            QString imageName = QString("image%1.tiff").arg(idx);
            QString cornerPath = QString("%1/%2.txt").arg(cornerDir).arg(imageName);

            QFile cornerFile(cornerPath);
            if (!cornerFile.open(QIODevice::ReadOnly | QIODevice::Text)) continue;

            QImage img(imgDir + "/" + imageName);
            if (img.isNull()) { cornerFile.close(); continue; }

            QImage drawImg = img.convertToFormat(QImage::Format_RGB888);
            QPainter painter(&drawImg);
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(255, 0, 0));

            QTextStream ts(&cornerFile);
            while (!ts.atEnd()) {
                QString line = ts.readLine().trimmed();
                if (line.isEmpty()) continue;
                QStringList parts = line.split(' ', Qt::SkipEmptyParts);
                if (parts.size() >= 2) {
                    double u = parts[0].toDouble();
                    double v = parts[1].toDouble();
                    painter.drawEllipse(QPointF(u, v), 5, 5);
                }
            }
            painter.end();
            cornerFile.close();

            QString outPath = QString("%1/%2_image%3_corners.png").arg(vizDir).arg(prefix).arg(idx);
            drawImg.save(outPath);
            vizCount++;
        }
        msgs << QString("Saved %1 %2 visualizations to %3").arg(vizCount).arg(prefix).arg(vizDir);
    };

    visualize(dataDir + "/images/camera0", "./output/corners_handeye", 0, 14, "handeye");
    visualize(dataDir + "/internal_images/camera0", "./output/corners_internal", 0, 41, "internal");

    msgs << "Corner detection complete.";
    return msgs;
}

} // namespace corner_detection
