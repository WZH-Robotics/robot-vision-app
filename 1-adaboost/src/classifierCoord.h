#ifndef KCLASSIFIERCOORD_H
#define KCLASSIFIERCOORD_H

#include <vector>
#include "featureCoord.h"

class KWeakClassifierCoord
{
    KFeatureCoord   _F;         //feature
    double          _dThresh;   //threshold    +1 if F(input)*nPolar > dThresh*nPolar
    int             _nPolar;    //weight       -1 otherwise
    double          _dAlpha;

public:
    KWeakClassifierCoord(){ };
    int     operator()(const KVector& vX) const { return (_F(vX)*_nPolar > _dThresh*_nPolar ? +1 : -1); }
    KVector operator()(const std::vector<KVector>& lX);

    double  Train(const std::vector<KVector>& lX,
                  const KVector& vY, const KVector& vW,
                  const std::vector<KFeatureCoord>& lF);

    double  OptimalThreshold(const KVector& vGap,
                             const KVector& vY,
                             const KVector& vW);
};


class KStrongClassifierCoord : public std::vector<KWeakClassifierCoord>
{
    KVector     _vAlpha;

public:
    KStrongClassifierCoord(){ };
    int     operator()(const KVector& vX){ return Classify(vX); };
    KVector operator()(const std::vector<KVector>& lX){ return Classify(lX); }

    void    Train(const std::vector<KVector>& lX,
                  const KVector& vY,
                  const std::vector<KFeatureCoord>& lF,
                  int nTmax=10,
                  double dMinError=0.01);
    int     Classify(const KVector& vX);
    KVector Classify(const std::vector<KVector>& lX);

};

#endif // KWEAKCLASSIFIERCOORD_H
