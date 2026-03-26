#include "handeye_calibration.h"
#include "pose_estimation.h"
#include "conventionalaxxbsvdsolver.h"

#include <QFile>
#include <QTextStream>
#include <QDir>
#include <QDebug>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace handeye_calibration {

bool saveHandEyeResult(const QString& filePath, const HandEyeResult& result) {
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "Failed to open file for writing:" << filePath;
        return false;
    }
    QTextStream out(&file);
    out.setRealNumberPrecision(15);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out << result.X(i, j);
            if (j < 3) out << " ";
        }
        out << "\n";
    }
    file.close();
    return true;
}

static Vector3d rotToRvec(const Matrix3d& R) {
    double cosAngle = (R.trace() - 1.0) / 2.0;
    cosAngle = max(-1.0, min(1.0, cosAngle));
    double angle = acos(cosAngle);
    if (angle < 1e-10) return Vector3d::Zero();
    Matrix3d skew = (R - R.transpose()) / (2.0 * sin(angle));
    return Vector3d(skew(2,1), skew(0,2), skew(1,0)) * angle;
}

static Matrix3d rvecToRot(const Vector3d& rvec) {
    double angle = rvec.norm();
    if (angle < 1e-12) return Matrix3d::Identity();
    Vector3d axis = rvec / angle;
    Matrix3d K;
    K << 0, -axis(2), axis(1), axis(2), 0, -axis(0), -axis(1), axis(0), 0;
    return Matrix3d::Identity() + sin(angle) * K + (1 - cos(angle)) * K * K;
}

static Matrix3d orthogonalizeRotation(const Matrix3d& R) {
    JacobiSVD<Matrix3d> svd(R, ComputeFullU | ComputeFullV);
    Matrix3d Ro = svd.matrixU() * svd.matrixV().transpose();
    if (Ro.determinant() < 0) {
        Matrix3d S = Matrix3d::Identity(); S(2,2) = -1;
        Ro = svd.matrixU() * S * svd.matrixV().transpose();
    }
    return Ro;
}

struct SolveResult {
    Matrix4d Z;   // T_EE->cam
    Matrix4d W;   // T_base->pattern
    double avgError;
    string label;
};

// Solve hand-eye for a given set of robot and camera poses
static SolveResult solveHandEye(const vector<Matrix4d>& robotPoses,
                                 const vector<Matrix4d>& cameraPoses,
                                 const string& label) {
    int n = min((int)robotPoses.size(), (int)cameraPoses.size());
    SolveResult result;
    result.label = label;

    // Step 1: Rotation via AX=XB relative transforms (Andreff SVD)
    Poses robotRelative, cameraRelative;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            Matrix4d A_rel = robotPoses[i].inverse() * robotPoses[j];
            Matrix4d B_rel = cameraPoses[i].inverse() * cameraPoses[j];
            double traceR = A_rel.block<3,3>(0,0).trace();
            double cosAngle = max(-1.0, min(1.0, (traceR - 1.0) / 2.0));
            double angle = acos(cosAngle) * 180.0 / M_PI;
            if (angle < 3.0) continue;
            robotRelative.push_back(A_rel);
            cameraRelative.push_back(B_rel);
        }
    }

    cout << "[" << label << "] Relative pairs: " << robotRelative.size() << " / " << n*(n-1)/2 << endl;

    if (robotRelative.size() < 2) {
        result.avgError = 1e30;
        return result;
    }

    ConventionalAXXBSVDSolver solver(robotRelative, cameraRelative);
    Pose X_andreff = solver.SolveX();
    Matrix3d Rz = X_andreff.block<3,3>(0,0);

    // Step 2: World rotation from absolute poses
    // Correct equation: R_robot(i) * Z * C_cam(i) = W
    // So: Rw = Rb * Rz * Ra (NOT Ra^T)
    Matrix3d Rw_sum = Matrix3d::Zero();
    for (int i = 0; i < n; ++i) {
        Matrix3d Rb = robotPoses[i].block<3,3>(0,0);
        Matrix3d Ra = cameraPoses[i].block<3,3>(0,0);
        Rw_sum += Rb * Rz * Ra;
    }
    Matrix3d Rw = orthogonalizeRotation(Rw_sum / n);

    // Step 3: Translation from absolute poses
    // tw = Rb*(Rz*ta + tz) + tb => Rb*tz - tw = -(Rb*Rz*ta + tb)
    MatrixXd A_trans = MatrixXd::Zero(3 * n, 6);
    VectorXd b_trans = VectorXd::Zero(3 * n);
    for (int i = 0; i < n; ++i) {
        Matrix3d Rb = robotPoses[i].block<3,3>(0,0);
        Vector3d tb = robotPoses[i].block<3,1>(0,3);
        Vector3d ta = cameraPoses[i].block<3,1>(0,3);
        A_trans.block<3,3>(3*i, 0) = Rb;
        A_trans.block<3,3>(3*i, 3) = -Matrix3d::Identity();
        b_trans.segment<3>(3*i) = -(Rb * Rz * ta + tb);
    }
    VectorXd t_sol = A_trans.bdcSvd(ComputeThinU | ComputeThinV).solve(b_trans);
    Vector3d tz = t_sol.head<3>();
    Vector3d tw = t_sol.tail<3>();

    // Step 4: Nonlinear refinement (LM)
    VectorXd params(12);
    params.head<3>() = rotToRvec(Rz);
    params.segment<3>(3) = tz;
    params.segment<3>(6) = rotToRvec(Rw);
    params.tail<3>() = tw;

    double lambda = 1e-3;
    for (int iter = 0; iter < 200; ++iter) {
        Matrix3d Rz_c = rvecToRot(params.head<3>());
        Vector3d tz_c = params.segment<3>(3);
        Matrix3d Rw_c = rvecToRot(params.segment<3>(6));
        Vector3d tw_c = params.tail<3>();

        VectorXd residual(12 * n);
        MatrixXd J(12 * n, 12);
        J.setZero();

        for (int i = 0; i < n; ++i) {
            Matrix3d Rb = robotPoses[i].block<3,3>(0,0);
            Vector3d tb = robotPoses[i].block<3,1>(0,3);
            Matrix3d Ra = cameraPoses[i].block<3,3>(0,0);
            Vector3d ta = cameraPoses[i].block<3,1>(0,3);

            // Correct: R(i)*Z*C(i) = W
            // Rotation: Rb*Rz*Ra - Rw
            // Translation: Rb*(Rz*ta + tz) + tb - tw
            Matrix3d lhsR = Rb * Rz_c * Ra;
            Vector3d lhsT = Rb * (Rz_c * ta + tz_c) + tb;
            Matrix3d rhsR = Rw_c;
            Vector3d rhsT = tw_c;

            Map<VectorXd>(residual.data() + 12*i, 9) =
                Map<const VectorXd>(lhsR.data(), 9) - Map<const VectorXd>(rhsR.data(), 9);
            residual.segment<3>(12*i + 9) = lhsT - rhsT;

            // Jacobian w.r.t. rvec_z (params 0-2):
            // d(Rb*Rz*Ra)/drvec_z[c] = Rb * Rz * [e_c]_x * Ra
            // d(Rb*(Rz*ta+tz))/drvec_z[c] = Rb * Rz * [e_c]_x * ta
            for (int c = 0; c < 3; ++c) {
                Vector3d e = Vector3d::Zero(); e(c) = 1.0;
                Matrix3d dRz; dRz << 0,-e(2),e(1), e(2),0,-e(0), -e(1),e(0),0;
                Matrix3d dLhsR = Rb * Rz_c * dRz * Ra;
                Map<VectorXd>(J.data() + (12*i) + c*J.rows(), 9) =
                    Map<const VectorXd>(dLhsR.data(), 9);
                J.block<3,1>(12*i+9, c) = Rb * Rz_c * dRz * ta;
            }
            // Jacobian w.r.t. tz (params 3-5): only affects translation = Rb
            J.block<3,3>(12*i+9, 3) = Rb;
            // Jacobian w.r.t. rvec_w (params 6-8): d(-Rw)/drvec_w[c] = -(Rw*[e_c]_x)
            for (int c = 0; c < 3; ++c) {
                Vector3d e = Vector3d::Zero(); e(c) = 1.0;
                Matrix3d dRw; dRw << 0,-e(2),e(1), e(2),0,-e(0), -e(1),e(0),0;
                Matrix3d dRhsR = -(Rw_c * dRw);
                Map<VectorXd>(J.data() + (12*i) + (6+c)*J.rows(), 9) =
                    Map<const VectorXd>(dRhsR.data(), 9);
            }
            // Jacobian w.r.t. tw (params 9-11): -I
            J.block<3,3>(12*i+9, 9) = -Matrix3d::Identity();
        }

        double cost = residual.squaredNorm();
        MatrixXd JtJ = J.transpose() * J;
        VectorXd Jtr = -J.transpose() * residual;
        MatrixXd damped = JtJ + lambda * MatrixXd(JtJ.diagonal().asDiagonal());
        VectorXd delta = damped.ldlt().solve(Jtr);
        VectorXd params_new = params + delta;

        // Evaluate new cost
        Matrix3d Rz_n = rvecToRot(params_new.head<3>());
        Vector3d tz_n = params_new.segment<3>(3);
        Matrix3d Rw_n = rvecToRot(params_new.segment<3>(6));
        Vector3d tw_n = params_new.tail<3>();
        double newCost = 0;
        for (int i = 0; i < n; ++i) {
            Matrix3d Rb = robotPoses[i].block<3,3>(0,0);
            Vector3d tb = robotPoses[i].block<3,1>(0,3);
            Matrix3d Ra = cameraPoses[i].block<3,3>(0,0);
            Vector3d ta = cameraPoses[i].block<3,1>(0,3);
            Matrix3d lR = Rb * Rz_n * Ra;
            Vector3d lT = Rb * (Rz_n * ta + tz_n) + tb;
            newCost += (lR - Rw_n).squaredNorm() + (lT - tw_n).squaredNorm();
        }

        if (newCost < cost) {
            params = params_new;
            lambda *= 0.1;
            if (abs(cost - newCost) < 1e-14 * cost) break;
        } else {
            lambda *= 10.0;
            if (lambda > 1e20) break;
        }
    }

    Rz = orthogonalizeRotation(rvecToRot(params.head<3>()));
    tz = params.segment<3>(3);
    Rw = orthogonalizeRotation(rvecToRot(params.segment<3>(6)));
    tw = params.tail<3>();

    result.Z = Matrix4d::Identity();
    result.Z.block<3,3>(0,0) = Rz;
    result.Z.block<3,1>(0,3) = tz;
    result.W = Matrix4d::Identity();
    result.W.block<3,3>(0,0) = Rw;
    result.W.block<3,1>(0,3) = tw;

    // Verification & per-pose errors
    vector<double> poseErrors(n);
    double totalError = 0;
    for (int i = 0; i < n; ++i) {
        poseErrors[i] = (robotPoses[i] * result.Z * cameraPoses[i] - result.W).norm();
        totalError += poseErrors[i];
        cout << "[" << label << "] Pose " << i << " error: " << poseErrors[i] << endl;
    }
    result.avgError = totalError / n;
    cout << "[" << label << "] Avg error: " << result.avgError
         << ", ||tz||=" << tz.norm() << " mm" << endl;
    cout << "[" << label << "] Z (T_EE->cam):\n" << result.Z << endl;
    cout << "[" << label << "] W (T_base->pattern):\n" << result.W << endl;

    return result;
}

// Solve with outlier rejection: remove poses with error > 2*median, re-solve
static SolveResult solveHandEyeRobust(const vector<Matrix4d>& robotPoses,
                                       const vector<Matrix4d>& cameraPoses,
                                       const string& label) {
    SolveResult res = solveHandEye(robotPoses, cameraPoses, label);
    int n = min((int)robotPoses.size(), (int)cameraPoses.size());
    if (n < 4 || res.avgError > 1e20) return res;

    // Compute per-pose errors
    vector<pair<double,int>> errors;
    for (int i = 0; i < n; ++i) {
        double err = (robotPoses[i] * res.Z * cameraPoses[i] - res.W).norm();
        errors.push_back({err, i});
    }

    // Find median error
    vector<double> sortedErrs;
    for (auto& e : errors) sortedErrs.push_back(e.first);
    sort(sortedErrs.begin(), sortedErrs.end());
    double median = sortedErrs[sortedErrs.size() / 2];
    double threshold = max(median * 2.5, 5.0);

    // Filter inliers
    vector<Matrix4d> robotInliers, cameraInliers;
    for (int i = 0; i < n; ++i) {
        if (errors[i].first <= threshold) {
            robotInliers.push_back(robotPoses[i]);
            cameraInliers.push_back(cameraPoses[i]);
        } else {
            cout << "[" << label << "] OUTLIER: pose " << i
                 << " error=" << errors[i].first << " > threshold=" << threshold << endl;
        }
    }

    if ((int)robotInliers.size() < n && (int)robotInliers.size() >= 3) {
        cout << "[" << label << "] Re-solving with " << robotInliers.size()
             << "/" << n << " inliers" << endl;
        SolveResult resInlier = solveHandEye(robotInliers, cameraInliers, label + "_inlier");
        if (resInlier.avgError < res.avgError) {
            return resInlier;
        }
    }

    return res;
}

QStringList runHandEyeCalibration() {
    QStringList msgs;

    vector<Matrix4d> cameraPoses;
    if (!pose_estimation::loadTransformationMatrices("./output/camera_transforms.txt", cameraPoses)) {
        msgs << "ERROR: Failed to load camera_transforms.txt.";
        return msgs;
    }

    vector<Matrix4d> robotPosesRaw;
    if (!pose_estimation::loadTransformationMatrices("./output/robot_transforms.txt", robotPosesRaw)) {
        msgs << "ERROR: Failed to load robot_transforms.txt.";
        return msgs;
    }

    int n = min((int)cameraPoses.size(), (int)robotPosesRaw.size());
    msgs << QString("Loaded %1 pose pairs.").arg(n);
    if (n < 2) { msgs << "ERROR: Need >= 2 pairs."; return msgs; }

    // Convention A: robot_cali.txt contains T_base->EE (standard)
    msgs << "=== Convention A: robot poses = T_base->EE ===";
    SolveResult resA = solveHandEyeRobust(robotPosesRaw, cameraPoses, "T_base->EE");

    // Convention B: robot_cali.txt contains T_EE->base (inverse)
    vector<Matrix4d> robotPosesInv(n);
    for (int i = 0; i < n; ++i) robotPosesInv[i] = robotPosesRaw[i].inverse();

    msgs << "=== Convention B: robot poses = T_EE->base (using inverse) ===";
    SolveResult resB = solveHandEyeRobust(robotPosesInv, cameraPoses, "T_EE->base(inv)");

    // Pick best: prefer smaller ||tz|| if both have reasonable error
    SolveResult best = resA;
    QString chosenConv = "A (T_base->EE)";
    double normA = resA.Z.block<3,1>(0,3).norm();
    double normB = resB.Z.block<3,1>(0,3).norm();

    cout << "\n=== Comparison ===" << endl;
    cout << "Conv A: ||tz||=" << normA << " mm, avg_err=" << resA.avgError << endl;
    cout << "Conv B: ||tz||=" << normB << " mm, avg_err=" << resB.avgError << endl;

    if (normB < normA && resB.avgError < resA.avgError * 3.0) {
        best = resB;
        chosenConv = "B (T_EE->base inverted)";
    }

    msgs << QString("Chosen convention: %1").arg(chosenConv);
    cout << "Chosen: " << chosenConv.toStdString() << endl;

    // Output results
    msgs << "\nHand-Eye Z (T_EE->cam):";
    for (int i = 0; i < 4; ++i) {
        QString row;
        for (int j = 0; j < 4; ++j) {
            row += QString::number(best.Z(i, j), 'f', 6);
            if (j < 3) row += "  ";
        }
        msgs << row;
    }
    msgs << "\nWorld W (T_base->pattern):";
    for (int i = 0; i < 4; ++i) {
        QString row;
        for (int j = 0; j < 4; ++j) {
            row += QString::number(best.W(i, j), 'f', 6);
            if (j < 3) row += "  ";
        }
        msgs << row;
    }
    msgs << QString("Average verification error: %1").arg(best.avgError);
    msgs << QString("||tz|| = %1 mm").arg(best.Z.block<3,1>(0,3).norm());

    // Save
    QDir().mkpath("./output");
    HandEyeResult hr; hr.X = best.Z;
    saveHandEyeResult("./output/handeye_result.txt", hr);
    msgs << "Saved handeye_result.txt";

    HandEyeResult wr; wr.X = best.W;
    saveHandEyeResult("./output/world_transform.txt", wr);
    msgs << "Saved world_transform.txt";

    msgs << "Hand-Eye calibration complete.";
    return msgs;
}

} // namespace handeye_calibration
