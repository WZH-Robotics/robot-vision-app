#ifndef HANDEYE_CALIBRATION_H
#define HANDEYE_CALIBRATION_H

#include <QString>
#include <QStringList>
#include <Eigen/Dense>

namespace handeye_calibration {

struct HandEyeResult {
    Eigen::Matrix4d X;  // 4x4 hand-eye transformation
};

bool saveHandEyeResult(const QString& filePath, const HandEyeResult& result);
QStringList runHandEyeCalibration();

} // namespace handeye_calibration

#endif // HANDEYE_CALIBRATION_H
