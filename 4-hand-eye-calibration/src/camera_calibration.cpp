#include "camera_calibration.h"
#include "corner_detection.h"
#include "kfc.h"

#include <QFile>
#include <QTextStream>
#include <QDir>
#include <QDebug>
#include <fstream>
#include <iostream>

using namespace std;
using namespace Eigen;

namespace camera_calibration {

vector<pair<double, double>> generateModelPoints(
    double squareHeight, double squareWidth, int numRows, int numCols, int nFeature) {

    vector<pair<double, double>> modelPoints;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            double x = j * squareWidth;
            double y = i * squareHeight;
            modelPoints.push_back({x, y});
        }
    }

    while (static_cast<int>(modelPoints.size()) < nFeature) {
        modelPoints.push_back({0.0, 0.0});
    }
    if (static_cast<int>(modelPoints.size()) > nFeature) {
        modelPoints.resize(nFeature);
    }

    return modelPoints;
}

vector<vector<pair<double, double>>> readCornerFiles(
    const string& dir, int nImg, int nFeature) {

    vector<vector<pair<double, double>>> pointzip;

    for (int i = 0; i < nImg; ++i) {
        string path = dir + "image" + to_string(i) + ".tiff.txt";
        ifstream file(path);
        if (!file.is_open()) continue; // Skip missing files (failed detection)

        vector<pair<double, double>> imagePoints;
        double x, y;
        while (file >> x >> y) {
            imagePoints.push_back({x, y});
        }

        // Only include if we got the expected number of corners
        if (static_cast<int>(imagePoints.size()) != nFeature) {
            qDebug() << "  Skipping" << QString::fromStdString(path)
                     << "- got" << imagePoints.size() << "expected" << nFeature;
            continue;
        }

        pointzip.push_back(imagePoints);
    }

    return pointzip;
}

bool saveCalibrationResult(const QString& filePath, const CalibrationResult& result) {
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "Failed to open file for writing:" << filePath;
        return false;
    }

    QTextStream out(&file);
    out.setRealNumberPrecision(15);

    // Intrinsic matrix (3x3)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out << result.intrinsicMatrix(i, j);
            if (j < 2) out << " ";
        }
        out << "\n";
    }

    // Distortion coefficients
    for (int i = 0; i < result.distortionCoefficients.size(); ++i) {
        out << result.distortionCoefficients(i);
        if (i < result.distortionCoefficients.size() - 1) out << " ";
    }
    out << "\n";

    // Error
    out << result.error << "\n";

    file.close();
    return true;
}

bool loadCalibrationResult(const QString& filePath, CalibrationResult& result) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "Failed to open file for reading:" << filePath;
        return false;
    }

    QTextStream in(&file);

    // Read intrinsic matrix (3x3)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            in >> result.intrinsicMatrix(i, j);
        }
    }

    // Read distortion coefficients (5 values)
    result.distortionCoefficients.resize(5);
    for (int i = 0; i < 5; ++i) {
        in >> result.distortionCoefficients(i);
    }

    // Read error
    in >> result.error;

    file.close();
    return true;
}

QStringList runCameraCalibration(const QString& calibObjectFile) {
    QStringList msgs;

    // Read chessboard specs
    corner_detection::ChessboardSpecs specs = corner_detection::readChessboardSpecs(calibObjectFile);
    if (specs.height == 0 || specs.width == 0) {
        msgs << "ERROR: Failed to load chessboard specs.";
        return msgs;
    }

    int nFeature = specs.height * specs.width;
    int nImg_max = 42;
    int nItr_int = 100;

    // Generate model points
    vector<pair<double, double>> modelPoints =
        generateModelPoints(specs.mmHeight, specs.mmWidth, specs.height, specs.width, nFeature);
    msgs << QString("Generated %1 model points.").arg(modelPoints.size());

    // Read internal corner files (only successfully detected images)
    vector<vector<pair<double, double>>> pointzip_int =
        readCornerFiles("./output/corners_internal/", nImg_max, nFeature);
    int nImg_int = (int)pointzip_int.size();
    if (nImg_int < 3) {
        msgs << QString("ERROR: Only %1 valid corner files found. Need at least 3.").arg(nImg_int);
        return msgs;
    }
    pointzip_int.push_back(modelPoints);
    msgs << QString("Loaded %1 valid internal image corner sets (out of %2).").arg(nImg_int).arg(nImg_max);

    // Zhang's calibration
    msgs << "Running Zhang's calibration...";
    KVector vX = cameracalib::ZhangsCalibration(pointzip_int, nImg_int, nFeature, nItr_int);

    // Extract intrinsic parameters
    CalibrationResult result;
    result.intrinsicMatrix = Matrix3d::Zero();
    result.intrinsicMatrix(0, 0) = vX[0];   // alpha (fx)
    result.intrinsicMatrix(0, 1) = 0.0;
    result.intrinsicMatrix(1, 1) = vX[1];   // beta (fy)
    result.intrinsicMatrix(0, 2) = vX[2];   // u0
    result.intrinsicMatrix(1, 2) = vX[3];   // v0
    result.intrinsicMatrix(2, 2) = 1.0;

    double dK1 = vX[4];
    double dK2 = vX[5];
    result.error = vX[vX.Size() - 1];

    result.distortionCoefficients.resize(5);
    result.distortionCoefficients << dK1, dK2, 0.0, 0.0, 0.0;

    msgs << QString("Intrinsic: fx=%1, fy=%2, cx=%3, cy=%4")
                .arg(result.intrinsicMatrix(0, 0))
                .arg(result.intrinsicMatrix(1, 1))
                .arg(result.intrinsicMatrix(0, 2))
                .arg(result.intrinsicMatrix(1, 2));
    msgs << QString("Distortion: k1=%1, k2=%2").arg(dK1).arg(dK2);
    msgs << QString("Calibration error: %1").arg(result.error);

    // Save result
    QDir().mkpath("./output");
    if (saveCalibrationResult("./output/camera_calibration.txt", result)) {
        msgs << "Saved camera_calibration.txt";
    } else {
        msgs << "ERROR: Failed to save camera_calibration.txt";
    }

    msgs << "Camera calibration complete.";
    return msgs;
}

} // namespace camera_calibration
