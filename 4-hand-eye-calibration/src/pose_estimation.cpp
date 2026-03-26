#include "pose_estimation.h"
#include "camera_calibration.h"
#include "corner_detection.h"
#include "kfc.h"

#include <QFile>
#include <QTextStream>
#include <QDir>
#include <QDebug>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace Eigen;

namespace pose_estimation {

// Rodrigues: rotation vector -> rotation matrix
Matrix3d rodrigues(const Vector3d& rvec) {
    double angle = rvec.norm();
    if (angle < 1e-12) return Matrix3d::Identity();
    Vector3d axis = rvec / angle;
    Matrix3d K;
    K << 0, -axis(2), axis(1),
         axis(2), 0, -axis(0),
        -axis(1), axis(0), 0;
    return Matrix3d::Identity() + sin(angle) * K + (1 - cos(angle)) * K * K;
}

// Rotation matrix -> Rodrigues vector
Vector3d invRodrigues(const Matrix3d& R) {
    double cosAngle = (R.trace() - 1.0) / 2.0;
    cosAngle = max(-1.0, min(1.0, cosAngle));
    double angle = acos(cosAngle);
    if (angle < 1e-12) return Vector3d::Zero();
    Matrix3d skew = (R - R.transpose()) / (2.0 * sin(angle));
    Vector3d axis(skew(2,1), skew(0,2), skew(1,0));
    return axis * angle;
}

// Planar PnP using HW3 approach + Gauss-Newton refinement:
// 1) Pixel-coord homography via DLT with NormalizeCoordinates (kfc)
// 2) Decompose: [r1 r2 t] = K^-1 * H, SVD orthogonalization
// 3) Gauss-Newton nonlinear refinement (minimize reprojection error)
// Iterative undistortion: given distorted normalized coords, find undistorted
static Vector2d iterativeUndistort(double xd, double yd, double k1, double k2) {
    double x = xd, y = yd;
    for (int iter = 0; iter < 20; ++iter) {
        double r2 = x * x + y * y;
        double r4 = r2 * r2;
        double radial = 1.0 + k1 * r2 + k2 * r4;
        x = xd / radial;
        y = yd / radial;
    }
    return Vector2d(x, y);
}

bool solvePnPPlanar(const vector<Vector2d>& modelPoints2D,
                    const vector<Vector2d>& imagePoints,
                    const Matrix3d& cameraMatrix,
                    const VectorXd& distCoeffs,
                    Matrix4d& pose) {
    int n = modelPoints2D.size();
    if (n < 4) return false;

    double fx = cameraMatrix(0,0), fy = cameraMatrix(1,1);
    double cx = cameraMatrix(0,2), cy = cameraMatrix(1,2);
    double k1 = distCoeffs.size() > 0 ? distCoeffs[0] : 0.0;
    double k2 = distCoeffs.size() > 1 ? distCoeffs[1] : 0.0;

    // Undistort image points before DLT
    vector<Vector2d> undistortedPts(n);
    for (int i = 0; i < n; ++i) {
        double xn = (imagePoints[i](0) - cx) / fx;
        double yn = (imagePoints[i](1) - cy) / fy;
        Vector2d corrected = iterativeUndistort(xn, yn, k1, k2);
        undistortedPts[i] = Vector2d(corrected(0) * fx + cx, corrected(1) * fy + cy);
    }

    // Build KMatrix for model (3 x n) and image (3 x n) in pixel coords
    KMatrix mM(3, n), mF(3, n);
    for (int i = 0; i < n; ++i) {
        mM[0][i] = modelPoints2D[i](0);
        mM[1][i] = modelPoints2D[i](1);
        mM[2][i] = 1.0;

        mF[0][i] = undistortedPts[i](0);
        mF[1][i] = undistortedPts[i](1);
        mF[2][i] = 1.0;
    }

    // --- EstimateHomography (HW3 approach) ---
    KMatrix mFn, mMn;
    KMatrix mTf = cameracalib::NormalizeCoordinates(mF, mFn);
    KMatrix mTm = cameracalib::NormalizeCoordinates(mM, mMn);

    KMatrix mA_dlt;
    for (int i = 0; i < n; ++i) {
        KVector vM(mMn[0][i], mMn[1][i], 1.0);
        mA_dlt ^= KVector(0.0, 0.0, 0.0).Tail(-vM).Tail(vM * mFn[1][i]).Tr();
        mA_dlt ^= vM.Tail(KVector(0.0, 0.0, 0.0)).Tail(vM * (-mFn[0][i])).Tr();
        mA_dlt ^= (vM * (-mFn[1][i])).Tail(vM * mFn[0][i]).Tail(KVector(0.0, 0.0, 0.0)).Tr();
    }

    KMatrix mU_svd, mV_svd;
    KVector vD, vH;
    mA_dlt.SVD(mU_svd, vD, mV_svd);
    vH = mV_svd.Column(mV_svd.Col() - 1);

    KMatrix mHn = (vH.Cut(0, 2).Tr() ^ vH.Cut(3, 5).Tr() ^ vH.Cut(6, 8).Tr());
    KMatrix mH = ~mTf * mHn * mTm;
    mH /= mH[2][2];

    // --- ExtractExtrinsics (HW3 approach): [r1 r2 t] = K^-1 * H ---
    KMatrix mK(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            mK[i][j] = cameraMatrix(i, j);

    double dSize;
    KVector vR1 = mK.Iv() * mH.Column(0); vR1.Normalized(_SIZE_NORMALIZE, &dSize);
    KVector vR2 = mK.Iv() * mH.Column(1); vR2.Normalized(_SIZE_NORMALIZE);
    KVector vR3 = vR1.Skew() * vR2;
    KVector vT  = (1.0 / dSize) * mK.Iv() * mH.Column(2);

    // Try both decompositions: original [r1,r2,r3,t] and alternative [-r1,-r2,r3,-t]
    // Run GN on each, keep the one with lower reprojection error
    double bestReprojError = 1e30;
    Matrix3d bestR;
    Vector3d bestT;

    for (int attempt = 0; attempt < 2; ++attempt) {
        KVector aR1 = (attempt == 0) ? vR1 : vR1 * (-1.0);
        KVector aR2 = (attempt == 0) ? vR2 : vR2 * (-1.0);
        KVector aR3 = aR1.Skew() * aR2;
        KVector aT  = (attempt == 0) ? vT : vT * (-1.0);

        // SVD orthogonalization
        KMatrix mR = (aR1 | aR2 | aR3);
        KMatrix mU2, mV2;
        KVector vD2;
        mR.SVD(mU2, vD2, mV2);
        mR = mU2 * mV2.Tr();

        // Convert to Eigen
        Matrix3d R;
        Vector3d t;
        for (int ii = 0; ii < 3; ++ii) {
            for (int jj = 0; jj < 3; ++jj)
                R(ii, jj) = mR[ii][jj];
            t(ii) = aT[ii];
        }

        // Ensure proper rotation (det = +1)
        if (R.determinant() < 0) {
            JacobiSVD<Matrix3d> svdR(R, ComputeFullU | ComputeFullV);
            Matrix3d U = svdR.matrixU();
            Matrix3d V = svdR.matrixV();
            Matrix3d S = Matrix3d::Identity();
            S(2,2) = (U * V.transpose()).determinant();
            R = U * S * V.transpose();
        }

        // Skip if camera behind pattern
        if (t(2) < 0) continue;

        // --- Levenberg-Marquardt nonlinear refinement ---
        Vector3d rvec = invRodrigues(R);
        double lambda = 1e-3;
        double prevCost = 1e30;

        for (int iter = 0; iter < 100; ++iter) {
            Matrix3d Rc = rodrigues(rvec);
            VectorXd residual(2 * n);
            MatrixXd J(2 * n, 6);

            for (int i = 0; i < n; ++i) {
                Vector3d P(modelPoints2D[i](0), modelPoints2D[i](1), 0.0);
                Vector3d Pc = Rc * P + t;
                double X = Pc(0), Y = Pc(1), Z = Pc(2);
                if (abs(Z) < 1e-6) Z = 1e-6;
                double invZ = 1.0 / Z;
                double invZ2 = invZ * invZ;

                // Normalized coords
                double xn = X * invZ, yn = Y * invZ;
                double r2 = xn * xn + yn * yn;
                double r4 = r2 * r2;
                double radial = 1.0 + k1 * r2 + k2 * r4;

                // Distorted projection
                double ud = fx * xn * radial + cx;
                double vd = fy * yn * radial + cy;

                residual(2*i)   = ud - imagePoints[i](0);
                residual(2*i+1) = vd - imagePoints[i](1);

                // Jacobian of distorted projection w.r.t. Pc
                double dr2_dxn = 2.0 * xn, dr2_dyn = 2.0 * yn;
                double drad_dxn = k1 * dr2_dxn + k2 * 2.0 * r2 * dr2_dxn;
                double drad_dyn = k1 * dr2_dyn + k2 * 2.0 * r2 * dr2_dyn;

                // du/dPc, dv/dPc via chain rule
                double dxn_dX = invZ, dxn_dZ = -X * invZ2;
                double dyn_dY = invZ, dyn_dZ = -Y * invZ2;

                double du_dX = fx * (radial * dxn_dX + xn * drad_dxn * dxn_dX);
                double du_dY = fx * xn * drad_dyn * dyn_dY;
                double du_dZ = fx * (radial * dxn_dZ + xn * (drad_dxn * dxn_dZ + drad_dyn * dyn_dZ));
                double dv_dX = fy * yn * drad_dxn * dxn_dX;
                double dv_dY = fy * (radial * dyn_dY + yn * drad_dyn * dyn_dY);
                double dv_dZ = fy * (radial * dyn_dZ + yn * (drad_dxn * dxn_dZ + drad_dyn * dyn_dZ));

                Matrix<double, 2, 3> dproj;
                dproj << du_dX, du_dY, du_dZ,
                         dv_dX, dv_dY, dv_dZ;

                Vector3d RP = Rc * P;
                Matrix3d skewRP;
                skewRP << 0, -RP(2), RP(1),
                          RP(2), 0, -RP(0),
                         -RP(1), RP(0), 0;

                J.block<2,3>(2*i, 0) = dproj * (-skewRP);
                J.block<2,3>(2*i, 3) = dproj;
            }

            double cost = residual.squaredNorm();

            // LM damping: (J^T J + lambda * diag(J^T J)) * delta = -J^T * r
            MatrixXd JtJ = J.transpose() * J;
            VectorXd Jtr = -J.transpose() * residual;
            MatrixXd damped = JtJ + lambda * MatrixXd(JtJ.diagonal().asDiagonal());
            VectorXd delta = damped.ldlt().solve(Jtr);

            // Trial step
            Vector3d rvec_new = rvec + delta.head<3>();
            Vector3d t_new = t + delta.tail<3>();

            // Evaluate new cost (with distortion)
            Matrix3d Rc_new = rodrigues(rvec_new);
            double newCost = 0;
            for (int i = 0; i < n; ++i) {
                Vector3d P(modelPoints2D[i](0), modelPoints2D[i](1), 0.0);
                Vector3d Pc = Rc_new * P + t_new;
                double xn = Pc(0) / Pc(2), yn = Pc(1) / Pc(2);
                double r2 = xn*xn + yn*yn;
                double rad = 1.0 + k1*r2 + k2*r2*r2;
                double u = fx * xn * rad + cx;
                double v = fy * yn * rad + cy;
                double du = u - imagePoints[i](0);
                double dv = v - imagePoints[i](1);
                newCost += du*du + dv*dv;
            }

            if (newCost < cost) {
                rvec = rvec_new;
                t = t_new;
                lambda *= 0.1;
                if (abs(cost - newCost) < 1e-12) break;
                prevCost = newCost;
            } else {
                lambda *= 10.0;
                if (lambda > 1e16) break;
            }
        }

        R = rodrigues(rvec);

        // Compute final reprojection error (RMS pixels, with distortion)
        double reproj = 0;
        for (int i = 0; i < n; ++i) {
            Vector3d P(modelPoints2D[i](0), modelPoints2D[i](1), 0.0);
            Vector3d Pc = R * P + t;
            if (Pc(2) <= 0) { reproj = 1e30; break; }
            double xn = Pc(0) / Pc(2), yn = Pc(1) / Pc(2);
            double r2 = xn*xn + yn*yn;
            double rad = 1.0 + k1*r2 + k2*r2*r2;
            double u = fx * xn * rad + cx;
            double v = fy * yn * rad + cy;
            reproj += (u - imagePoints[i](0)) * (u - imagePoints[i](0))
                    + (v - imagePoints[i](1)) * (v - imagePoints[i](1));
        }
        reproj = sqrt(reproj / n);

        cout << "  Attempt " << attempt << ": reproj=" << reproj
             << " trace(R)=" << R.trace() << " tz=" << t(2) << endl;

        if (reproj < bestReprojError) {
            bestReprojError = reproj;
            bestR = R;
            bestT = t;
        }
    }

    // Reject if reprojection error too large (bad PnP)
    if (bestReprojError > 10.0) {
        cout << "  REJECTED: reproj=" << bestReprojError << " too large" << endl;
        return false;
    }

    pose = Matrix4d::Identity();
    pose.block<3, 3>(0, 0) = bestR;
    pose.block<3, 1>(0, 3) = bestT;

    cout << "Pose (best reproj=" << bestReprojError << "):\n" << pose << endl;

    return true;
}

vector<IndexedCorners> readCornerFilesIndexed(const string& dir, int nImg, int nFeature) {
    vector<IndexedCorners> result;
    for (int i = 0; i < nImg; ++i) {
        string path = dir + "image" + to_string(i) + ".tiff.txt";
        ifstream file(path);
        if (!file.is_open()) continue;

        vector<pair<double, double>> pts;
        double x, y;
        while (file >> x >> y) {
            pts.push_back({x, y});
        }
        if ((int)pts.size() != nFeature) continue;

        IndexedCorners ic;
        ic.imageIndex = i;
        ic.points = pts;
        result.push_back(ic);
    }
    return result;
}

vector<Matrix4d> readTransformationMatrices(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return {};
    }

    int numMatrices;
    file >> numMatrices;

    vector<Matrix4d> transformations;
    transformations.reserve(numMatrices);

    Matrix4d matrix;
    for (int i = 0; i < numMatrices; i++) {
        if (file >> matrix(0,0) >> matrix(0,1) >> matrix(0,2) >> matrix(0,3)
            >> matrix(1,0) >> matrix(1,1) >> matrix(1,2) >> matrix(1,3)
            >> matrix(2,0) >> matrix(2,1) >> matrix(2,2) >> matrix(2,3)
            >> matrix(3,0) >> matrix(3,1) >> matrix(3,2) >> matrix(3,3)) {
            transformations.push_back(matrix);
        } else {
            cerr << "Error reading matrix at index " << i << endl;
            break;
        }
    }

    file.close();
    return transformations;
}

bool saveTransformationMatrices(const QString& filePath,
                                const vector<Matrix4d>& matrices) {
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "Failed to open file for writing:" << filePath;
        return false;
    }

    QTextStream out(&file);
    out.setRealNumberPrecision(15);

    out << matrices.size() << "\n";
    for (const auto& mat : matrices) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                out << mat(i, j);
                if (j < 3) out << " ";
            }
            out << "\n";
        }
    }

    file.close();
    return true;
}

bool loadTransformationMatrices(const QString& filePath,
                                vector<Matrix4d>& matrices) {
    string fn = filePath.toStdString();
    matrices = readTransformationMatrices(fn);
    return !matrices.empty();
}

QStringList runPoseEstimation(const QString& calibObjectFile,
                              const QString& robotCaliFile) {
    QStringList msgs;
    cout << "=== Pose Estimation Started ===" << endl;

    // Load camera calibration result
    camera_calibration::CalibrationResult calib;
    if (!camera_calibration::loadCalibrationResult("./output/camera_calibration.txt", calib)) {
        msgs << "ERROR: Failed to load camera_calibration.txt. Run Camera Calibration first.";
        return msgs;
    }
    msgs << "Loaded camera calibration result.";

    // Read chessboard specs for model points
    corner_detection::ChessboardSpecs specs = corner_detection::readChessboardSpecs(calibObjectFile);
    if (specs.height == 0 || specs.width == 0) {
        msgs << "ERROR: Failed to load chessboard specs.";
        return msgs;
    }

    int nFeature = specs.height * specs.width;
    int nImg = 15;

    // Generate 2D model points (planar, Z=0)
    vector<pair<double, double>> modelPointsPairs =
        camera_calibration::generateModelPoints(specs.mmHeight, specs.mmWidth,
                                                specs.height, specs.width, nFeature);

    vector<Vector2d> modelPoints2D;
    for (const auto& p : modelPointsPairs) {
        modelPoints2D.emplace_back(p.first, p.second);
    }

    // Read hand-eye corner files WITH image index tracking
    vector<IndexedCorners> indexedCorners =
        readCornerFilesIndexed("./output/corners_handeye/", nImg, nFeature);
    msgs << QString("Loaded %1 valid hand-eye corner sets.").arg(indexedCorners.size());

    // Compute absolute camera pose for each detected image
    struct IndexedPose {
        int imageIndex;
        Matrix4d pose;
    };
    vector<IndexedPose> cameraPoses;

    for (const auto& ic : indexedCorners) {
        vector<Vector2d> imgPts;
        for (const auto& p : ic.points) {
            imgPts.emplace_back(p.first, p.second);
        }

        Matrix4d camPose;
        if (solvePnPPlanar(modelPoints2D, imgPts, calib.intrinsicMatrix,
                           calib.distortionCoefficients, camPose)) {
            cameraPoses.push_back({ic.imageIndex, camPose});
            msgs << QString("Pose for image %1 computed.").arg(ic.imageIndex);
        } else {
            msgs << QString("ERROR: PnP failed for image %1").arg(ic.imageIndex);
        }
    }

    // Read robot transformations (absolute poses, indexed 0..14)
    vector<Matrix4d> robotTransformations = readTransformationMatrices(robotCaliFile.toStdString());
    msgs << QString("Loaded %1 robot transformations.").arg(robotTransformations.size());

    // For AX=ZB: save absolute camera poses and matched robot poses
    vector<Matrix4d> absoluteCameraPoses;
    vector<Matrix4d> matchedRobotPoses;

    for (const auto& cp : cameraPoses) {
        int idx = cp.imageIndex;
        if (idx >= (int)robotTransformations.size()) continue;

        absoluteCameraPoses.push_back(cp.pose);
        matchedRobotPoses.push_back(robotTransformations[idx]);

        msgs << QString("Matched image %1 -> robot pose %1.").arg(idx);
    }

    msgs << QString("Total matched poses: %1").arg(absoluteCameraPoses.size());
    cout << "Total matched poses: " << absoluteCameraPoses.size() << endl;

    // Save results
    QDir().mkpath("./output");
    if (saveTransformationMatrices("./output/camera_transforms.txt", absoluteCameraPoses)) {
        msgs << "Saved camera_transforms.txt (absolute poses)";
    } else {
        msgs << "ERROR: Failed to save camera_transforms.txt";
    }

    if (saveTransformationMatrices("./output/robot_transforms.txt", matchedRobotPoses)) {
        msgs << "Saved robot_transforms.txt (matched robot poses)";
    } else {
        msgs << "ERROR: Failed to save robot_transforms.txt";
    }

    msgs << "Pose estimation complete.";
    return msgs;
}

} // namespace pose_estimation
