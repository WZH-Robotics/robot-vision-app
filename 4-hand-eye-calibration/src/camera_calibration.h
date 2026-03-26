#ifndef CAMERA_CALIBRATION_H
#define CAMERA_CALIBRATION_H

#include <QString>
#include <QStringList>
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace camera_calibration {

struct CalibrationResult {
    Eigen::Matrix3d intrinsicMatrix;
    Eigen::VectorXd distortionCoefficients;  // [k1, k2, p1, p2, k3]
    double error;
};

std::vector<std::pair<double, double>> generateModelPoints(
    double squareHeight, double squareWidth, int numRows, int numCols, int nFeature);

std::vector<std::vector<std::pair<double, double>>> readCornerFiles(
    const std::string& dir, int nImg, int nFeature);

bool saveCalibrationResult(const QString& filePath, const CalibrationResult& result);
bool loadCalibrationResult(const QString& filePath, CalibrationResult& result);

QStringList runCameraCalibration(const QString& calibObjectFile);

} // namespace camera_calibration

#endif // CAMERA_CALIBRATION_H
