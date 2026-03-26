#ifndef ZHANGCALIBRATION_H
#define ZHANGCALIBRATION_H

#include "kfc.h"

#include <string>
#include <vector>

struct ZhangCalibResult {
    KMatrix mA;           // 3x3 intrinsic matrix
    KVector vX;           // optimized state vector (48-dim)
    double  dInitErf;
    double  dFinalErf;
    double  dRms;
};

// 모델 좌표 로딩 (156 points)
bool LoadModelPoints(const std::string& path, KPoint* psM, int nPoints);

// 이미지 좌표 로딩
bool LoadImagePoints(const std::string& path, KPoint* psF, int nPoints);

// DLT + SVD + 좌표 정규화로 호모그래피 추정
KMatrix EstimateHomography(const KMatrix& mF, const KMatrix& mM, int nPoints);

// V 제약행렬에 사용할 vij 벡터 생성 및 V 행렬 구성
void BuildVRow(const KMatrix& mH, int rowIndex, KMatrix& mV);

// SVD로 B 복원
KMatrix ExtractBMatrix(const KMatrix& mV);

// B → A (alpha, beta, u0, v0) 추출
KMatrix ExtractIntrinsics(const KMatrix& mB);

// 각 이미지 R, t + SVD 직교화
KHomogeneous ExtractExtrinsics(const KMatrix& mH, const KMatrix& mA);

// 48차원 상태벡터 구성
KVector BuildOptimizationVector(const KMatrix& mA, const KHomogeneous* lP, int nImages);

// --- Reprojection error function for Powell optimization ---
// (KOptima base class is in the professor's framework; only the Erf override is shown here)
class KCalibrationZhang : public KOptima {
    KMatrix _mModelRaw;        // 3x156 raw model points [X,Y,1]
    KMatrix _mImageRaw[7];     // 7 x (3x156) raw image points [u,v,1]

public:
    void setData(const KMatrix& mModel, const KMatrix mImages[], int nImages);
    double Erf(const KVector& vX);
};

#endif // ZHANGCALIBRATION_H
