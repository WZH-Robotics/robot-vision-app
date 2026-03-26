#ifndef FEATURECOORD_H
#define FEATURECOORD_H

#include "kfc.h"

class KFeatureCoord
{
    int _nAxis;

public:
    KFeatureCoord(int axis = _X_AXIS) : _nAxis(axis){ } //_X_AXIS=0, _Y_AXIS=1
    double  operator()(const KVector& vX) const {  return vX[_nAxis]; }
    KVector operator()(const std::vector<KVector>& lX) const
    {
        KVector vValue(lX.size());
        for(int i=0, ii=lX.size(); ii; i++, ii--)
            vValue[i]   = lX[i][_nAxis];
        return vValue;
    }
};

#endif // FEATURECOORD_H

