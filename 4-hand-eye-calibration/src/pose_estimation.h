#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

#include <QString>
#include <QStringList>
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace pose_estimation {

bool solvePnPPlanar(const std::vector<Eigen::Vector2d>& modelPoints2D,
                    const std::vector<Eigen::Vector2d>& imagePoints,
                    const Eigen::Matrix3d& cameraMatrix,
                    const Eigen::VectorXd& distCoeffs,
                    Eigen::Matrix4d& pose);

std::vector<Eigen::Matrix4d> readTransformationMatrices(const std::string& filename);

struct IndexedCorners {
    int imageIndex;
    std::vector<std::pair<double, double>> points;
};

std::vector<IndexedCorners> readCornerFilesIndexed(
    const std::string& dir, int nImg, int nFeature);

bool saveTransformationMatrices(const QString& filePath,
                                const std::vector<Eigen::Matrix4d>& matrices);
bool loadTransformationMatrices(const QString& filePath,
                                std::vector<Eigen::Matrix4d>& matrices);

QStringList runPoseEstimation(const QString& calibObjectFile,
                              const QString& robotCaliFile);

} // namespace pose_estimation

#endif // POSE_ESTIMATION_H
