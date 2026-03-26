#include "classifierCoord.h"
#include <limits>

using namespace std;

void KStrongClassifierCoord::Train(const vector<KVector>& lX,
                                   const KVector& vY,
                                   const vector<KFeatureCoord>& lF,
                                   int nTmax, double dMaxError)
{
    static int           nID;
    double               dError = 1000.0, dAlpha;
    int                  nNum = lX.size();
    KVector              vH(nNum), vW = KMatrix::Ones(nNum,1);
    KWeakClassifierCoord oClassifier;

    //clear previous state
    _vAlpha.Release();
    vector<KWeakClassifierCoord>::clear();

    //pre-allocate to avoid repeated Tailed() heap churn
    _vAlpha.Create(nTmax);
    vector<KWeakClassifierCoord>::reserve(nTmax);

    std::vector<double> vErrorLocal;
    vErrorLocal.reserve(nTmax);

    vW.Normalized(_UNITSUM_NORMALIZE);

    int nRounds = 0;
    for(int i=0; i<nTmax && (dError > dMaxError); i++)
    {
        dError = oClassifier.Train(lX, vY, vW, lF);

        //stop if weak classifier is no better than random
        if(dError >= 0.5)
            break;

        //clamp error to avoid log(0)
        if(dError < 1e-10)
            dError = 1e-10;

        //the weight of the classifier
        dAlpha = 0.5*log((1.0-dError) / dError);
        _vAlpha[i] = dAlpha;

        //add the weak classifier
        vector<KWeakClassifierCoord>::push_back(oClassifier);
        nRounds++;

        //update the weight of each misclassified sample
        for(int n=0; n<nNum; n++){
            vW[n] *= exp(-dAlpha*vY[n]*oClassifier(lX[n]));
        }

        //safe normalization: check sum to avoid NaN
        double dWsum = 0.0;
        for(int n=0; n<nNum; n++) dWsum += vW[n];
        if(dWsum > 0.0)
            vW.Normalized(_UNITSUM_NORMALIZE);
        else
            break;

        //validation (inline to avoid temporary KVector allocation)
        for(int n=0; n<nNum; n++)
            vH[n] = Classify(lX[n]);
        dError = 0.5 - 0.5 * (vY & vH) / (double)(vY.Dim());
        vErrorLocal.push_back(dError);
    }

    //trim _vAlpha to actual number of rounds
    if(nRounds > 0 && nRounds < nTmax)
    {
        KVector vAlphaTmp(nRounds);
        for(int i = 0; i < nRounds; i++)
            vAlphaTmp[i] = _vAlpha[i];
        _vAlpha = vAlphaTmp;
    }
    else if(nRounds == 0)
    {
        _vAlpha.Release();
    }

    //write error log
    if(!vErrorLocal.empty())
    {
        KVector vError((int)vErrorLocal.size());
        for(int i = 0; i < (int)vErrorLocal.size(); i++)
            vError[i] = vErrorLocal[i];
        vError.WriteText(KString::Format("./output/error_in_strongclassifier_train_%02d.txt",nID).Address());
    }
};


KVector KWeakClassifierCoord::operator()(const vector<KVector>& lX)
{
    KVector vYp(lX.size());

    for(int i=0, ii=lX.size(); ii; i++, ii--)
        vYp[i] = (*this)(lX[i]);

    return vYp;
}

int KStrongClassifierCoord::Classify(const KVector& vX)
{
    int     nNum = vector<KWeakClassifierCoord>::size();
    double  dH   = 0.0;

    for(int i=0, ii=nNum; ii; i++, ii--)
        dH += _vAlpha[i] * (*this)[i](vX);

    return _SIGN(dH);
}

KVector KStrongClassifierCoord::Classify(const vector<KVector>& lX)
{
    KVector vOut(lX.size());

    for(int n=0, nn=lX.size(); nn; n++, nn--)
        vOut[n] = (*this)( lX[n] );

    return vOut;
}

double KWeakClassifierCoord::Train(const vector<KVector>& lX, const KVector& vY,
                                   const KVector& vW, const vector<KFeatureCoord>& lF)
{
    KVector vGap;
    double  dThresh, dError, dH, dMin = 1000.0;
    int     nPolar;
    KVector vSample;
    KFeatureCoord  Fx(_X_AXIS), Fy(_Y_AXIS);

    vGap = vSample;

    for(int i=0, ii=lF.size(); ii; i++, ii--)
    {
        vGap    = lF[i](lX);
        dThresh = OptimalThreshold(vGap, vY, vW);

        //오차 계산
        dError  = 0.0;
        nPolar  = (dThresh > 0 ? +1 : -1);
        dThresh = (dThresh > 0 ? dThresh : -dThresh);

        for(int i=0, ii=lX.size(); ii; i++, ii--)
        {         
            dError += vW[i] * vY[i] * (vGap[i]*nPolar > dThresh*nPolar ? +1 : -1);
        }
        dError = 0.5 - 0.5*dError;

        //최소 오차의 Feature와 파라미터 설정
        if(dError < dMin)
        {
            dMin     = dError;
            _dThresh = dThresh;
            _nPolar  = nPolar;
            _F       = lF[i];
        }
    }

    return dMin;
}


double KWeakClassifierCoord::OptimalThreshold(const KVector& vGap, const KVector& vY, const KVector& vW)
{
    //Sorting in Ascending Order according to each feature value
    KVector     vGs(vGap), vYs(vY), vWs(vW);
    vector<int> lIndex;
    lIndex.reserve(vGs.Dim());

    vGs.QuickSorted(&lIndex);
    vYs.SortedBy(lIndex);
    vWs.SortedBy(lIndex);

    //Compute the cumulative sums of plus samples and minus samples
    int     nNum = vYs.Dim();
    KVector vCp(nNum), vCm(nNum);
    double  dTp  = 0.0, dTm  = 0.0;

    for(int i=0; i<nNum; i++)
    {
        if(vYs[i] > 0) dTp += vWs[i];
        else           dTm += vWs[i];

        vCp[i] = dTp;
        vCm[i] = dTm;
    }

    //Obtain the Optimal Threshold
    double dThresh=0.0, dEp, dEm, dMin = numeric_limits<double>::max();

    for(int i=0, ii=nNum-1; ii; i++, ii--)
    {
        dEp = dTm - vCm[i] + vCp[i];
        dEm = vCm[i] + dTp - vCp[i];   

        if(dEp < dMin)
        {
            dMin = dEp;
            dThresh = (vGs[i] + vGs[i+1]) * 0.5;
        }
        if(dEm < dMin)
        {
            dMin = dEm;
            dThresh = -(vGs[i] + vGs[i+1]) * 0.5; // 극성을 부호로 포함시킴
        }                                         // threshold = _ABS(dThresh)
    }                                             // polarity  = _SIGN(dThresh)

    return dThresh;
}

