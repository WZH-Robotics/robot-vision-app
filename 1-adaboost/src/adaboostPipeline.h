#ifndef ADABOOSTPIPELINE_H
#define ADABOOSTPIPELINE_H

#include "classifierCoord.h"
#include "featureCoord.h"
#include "kfc.h"

#include <vector>
#include <string>

struct ConfusionMatrix {
    int TP, FP, TN, FN;
    double error;
    double accuracy;
};

// CSV 파싱 → vector<KVector> + KVector labels
bool LoadSamplesFromCSV(const std::string& path,
                        std::vector<KVector>& lx,
                        KVector& vY);

// 10개 KFeatureCoord 생성
std::vector<KFeatureCoord> BuildFeatures();

// 혼동행렬(TP/FP/TN/FN) 계산
ConfusionMatrix EvaluateClassifier(KStrongClassifierCoord& classifier,
                                   const std::vector<KVector>& lx,
                                   const KVector& vY);

#endif // ADABOOSTPIPELINE_H
