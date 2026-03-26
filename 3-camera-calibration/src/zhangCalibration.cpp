#include "zhangCalibration.h"
#include <cstdio>
#include <cmath>
#include <iostream>

using namespace std;

bool LoadModelPoints(const std::string& path, KPoint* psM, int nPoints)
{
    FILE* fp = fopen(path.c_str(), "r");
    if (!fp) {
        cout << "ERROR: Cannot open " << path << endl;
        return false;
    }
    for (int i = 0; i < nPoints; i++)
        fscanf(fp, "%lf %lf\n", &psM[i]._dX, &psM[i]._dY);
    fclose(fp);
    return true;
}

bool LoadImagePoints(const std::string& path, KPoint* psF, int nPoints)
{
    FILE* fp = fopen(path.c_str(), "r");
    if (!fp) {
        cout << "ERROR: Cannot open " << path << endl;
        return false;
    }
    for (int i = 0; i < nPoints; i++)
        fscanf(fp, "%lf %lf\n", &psF[i]._dX, &psF[i]._dY);
    fclose(fp);
    return true;
}

KMatrix EstimateHomography(const KMatrix& mF, const KMatrix& mM, int nPoints)
{
    KMatrix mFn, mMn;
    KMatrix mTf, mTm;

    mTf = NormalizeCoordinates(mF, mFn);
    mTm = NormalizeCoordinates(mM, mMn);

    KMatrix mA;
    KVector vM;
    for (int i = 0, ii = nPoints; ii; i++, ii--)
    {
        vM  = KVector(mMn[0][i], mMn[1][i], 1.0);
        mA ^= KVector(0.0, 0.0, 0.0).Tail(-vM).Tail(vM*mFn[1][i]).Tr();
        mA ^= vM.Tail(KVector(0.0, 0.0, 0.0)).Tail(vM*(-mFn[0][i])).Tr();
        mA ^= (vM*(-mFn[1][i])).Tail(vM*mFn[0][i]).Tail(KVector(0.0, 0.0, 0.0)).Tr();
    }

    KMatrix mU_svd, mV_svd, mHn;
    KVector vD, vH;

    mA.SVD(mU_svd, vD, mV_svd);
    vH = mV_svd.Column(mV_svd.Col()-1);
    mHn = (vH.Cut(0,2).Tr() ^ vH.Cut(3,5).Tr() ^ vH.Cut(6,8).Tr());
    KMatrix mH = ~mTf * mHn * mTm;
    mH /= mH[2][2];

    return mH;
}

void BuildVRow(const KMatrix& mH, int rowIndex, KMatrix& mV)
{
    KVector vV11(6), vV12(6), vV22(6);

    vV11[0] = mH[0][0]*mH[0][0];
    vV11[1] = mH[0][0]*mH[1][0] + mH[1][0]*mH[0][0];
    vV11[2] = mH[1][0]*mH[1][0];
    vV11[3] = mH[2][0]*mH[0][0] + mH[0][0]*mH[2][0];
    vV11[4] = mH[2][0]*mH[1][0] + mH[1][0]*mH[2][0];
    vV11[5] = mH[2][0]*mH[2][0];

    vV12[0] = mH[0][0]*mH[0][1];
    vV12[1] = mH[0][0]*mH[1][1] + mH[1][0]*mH[0][1];
    vV12[2] = mH[1][0]*mH[1][1];
    vV12[3] = mH[2][0]*mH[0][1] + mH[0][0]*mH[2][1];
    vV12[4] = mH[2][0]*mH[1][1] + mH[1][0]*mH[2][1];
    vV12[5] = mH[2][0]*mH[2][1];

    vV22[0] = mH[0][1]*mH[0][1];
    vV22[1] = mH[0][1]*mH[1][1] + mH[1][1]*mH[0][1];
    vV22[2] = mH[1][1]*mH[1][1];
    vV22[3] = mH[2][1]*mH[0][1] + mH[0][1]*mH[2][1];
    vV22[4] = mH[2][1]*mH[1][1] + mH[1][1]*mH[2][1];
    vV22[5] = mH[2][1]*mH[2][1];

    mV.Place(2*rowIndex,   0, vV12.Tr());
    mV.Place(2*rowIndex+1, 0, (vV11-vV22).Tr());
}

KMatrix ExtractBMatrix(const KMatrix& mV)
{
    KMatrix mU_svd, mW_svd;
    KVector vD, vB;
    KMatrix mVcopy(mV);
    mVcopy.SVD(mU_svd, vD, mW_svd);
    vB = mW_svd.Column(mW_svd.Col()-1);

    KMatrix mB(3,3);
    mB[0][0] = vB[0]; mB[0][1] = vB[1]; mB[0][2] = vB[3];
    mB[1][0] = vB[1]; mB[1][1] = vB[2]; mB[1][2] = vB[4];
    mB[2][0] = vB[3]; mB[2][1] = vB[4]; mB[2][2] = vB[5];

    return mB;
}

KMatrix ExtractIntrinsics(const KMatrix& mB)
{
    KMatrix mA(3,3);
    double dLambda;

    mA[1][2] = (mB[0][1]*mB[0][2] - mB[0][0]*mB[1][2])
               / (mB[0][0]*mB[1][1] - mB[0][1]*mB[0][1]);
    dLambda  = mB[2][2] - (_SQR(mB[0][2]) + mA[1][2]*(mB[0][1]*mB[0][2]
                                                       - mB[0][0]*mB[1][2])) / mB[0][0];

    mA[0][0] = sqrt(dLambda / mB[0][0]);
    mA[1][1] = sqrt(dLambda*mB[0][0] / (mB[0][0]*mB[1][1] - _SQR(mB[0][1])));
    mA[0][2] = -mB[0][2] * _SQR(mA[0][0]) / dLambda;
    mA[2][2] = 1;

    return mA;
}

KHomogeneous ExtractExtrinsics(const KMatrix& mH, const KMatrix& mA)
{
    KMatrix mU_svd, mW_svd;
    KVector vD;

    double dSize;
    KVector vR1, vR2, vR3, vT;

    vR1 = mA.Iv() * mH.Column(0); vR1.Normalized(_SIZE_NORMALIZE, &dSize);
    vR2 = mA.Iv() * mH.Column(1); vR2.Normalized(_SIZE_NORMALIZE);
    vR3 = vR1.Skew() * vR2;
    vT  = (1.0/dSize) * mA.Iv() * mH.Column(2);

    KMatrix mR = (vR1 | vR2 | vR3);
    mR.SVD(mU_svd, vD, mW_svd);
    mR = mU_svd * mW_svd.Tr();

    return KHomogeneous(mR, vT);
}

KVector BuildOptimizationVector(const KMatrix& mA, const KHomogeneous* lP, int nImages)
{
    KVector vX;

    vX.Tailed(mA[0][0]); vX.Tailed(mA[1][1]); // alpha, beta
    vX.Tailed(mA[0][2]); vX.Tailed(mA[1][2]); // u0, v0
    vX.Tailed(KVector(0.0, 0.0));               // k1, k2

    for (int i = 0; i < nImages; i++)
        vX.Tailed(KHomogeneous(lP[i]).RT(_EULER));

    return vX;
}

// --- KCalibrationZhang: reprojection error for Powell optimization ---

void KCalibrationZhang::setData(const KMatrix& mModel, const KMatrix mImages[], int nImages)
{
    _mModelRaw = mModel;
    for(int i = 0; i < nImages && i < 7; i++)
        _mImageRaw[i] = mImages[i];
}

double KCalibrationZhang::Erf(const KVector& vX)
{
    // Extract intrinsic parameters
    int idx = 0;
    KMatrix mA(3,3);
    double dK1, dK2;

    mA[0][0] = vX[idx++];    // alpha
    mA[1][1] = vX[idx++];    // beta
    mA[0][2] = vX[idx++];    // u0
    mA[1][2] = vX[idx++];    // v0
    dK1      = vX[idx++];    // k1
    dK2      = vX[idx++];    // k2
    mA[2][2] = 1;

    int nPts = _mModelRaw.Col();  // 156
    double dError = 0;

    for(int i = 0; i < 7; i++)
    {
        // Extract extrinsic parameters (3 euler + 3 translation)
        KHomogeneous P(vX.Cut(idx, idx+5));
        idx += 6;

        KMatrix mR = P.R();
        KVector vT = P.t();

        for(int j = 0; j < nPts; j++)
        {
            // Model point on Z=0 plane
            KVector vM(3);
            vM[0] = _mModelRaw[0][j];
            vM[1] = _mModelRaw[1][j];
            vM[2] = 0.0;

            // World to camera: Xc = R * M + t
            KVector vXc = mR * vM + vT;

            // Normalized camera coordinates
            double xn = vXc[0] / vXc[2];
            double yn = vXc[1] / vXc[2];

            // Radial distortion
            double r2 = xn*xn + yn*yn;
            double dr = 1.0 + dK1*r2 + dK2*r2*r2;

            // Project to pixel with distortion
            double u_proj = mA[0][0] * xn * dr + mA[0][2];
            double v_proj = mA[1][1] * yn * dr + mA[1][2];

            // Observed image coordinates
            double u_obs = _mImageRaw[i][0][j];
            double v_obs = _mImageRaw[i][1][j];

            // Reprojection error
            dError += _SQR(u_proj - u_obs) + _SQR(v_proj - v_obs);
        }
    }

    return dError;
}
