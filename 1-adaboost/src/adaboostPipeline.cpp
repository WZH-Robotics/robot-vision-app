#include "adaboostPipeline.h"
#include <cstdio>

using namespace std;

bool LoadSamplesFromCSV(const std::string& path,
                        std::vector<KVector>& lx,
                        KVector& vY)
{
    lx.clear();
    lx.reserve(1024);

    KVector         vTmp(10);
    vector<double>  vLabels;
    int             nLabel;
    FILE*           fp = fopen(path.c_str(), "r");

    if (!fp) return false;

    char headerBuffer[256];
    fgets(headerBuffer, sizeof(headerBuffer), fp);

    while (fscanf(fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%d",
                  &vTmp[0], &vTmp[1], &vTmp[2], &vTmp[3], &vTmp[4],
                  &vTmp[5], &vTmp[6], &vTmp[7], &vTmp[8], &vTmp[9], &nLabel) == 11)
    {
        lx.push_back(vTmp);
        vLabels.push_back((double)nLabel);
    }
    fclose(fp);

    vY = KVector(vLabels.size());
    for (int i = 0; i < (int)vLabels.size(); i++)
        vY[i] = vLabels[i];

    return true;
}

std::vector<KFeatureCoord> BuildFeatures()
{
    std::vector<KFeatureCoord> lFeature;
    for (int i = 0; i < 10; i++)
        lFeature.push_back(KFeatureCoord(i));
    return lFeature;
}

ConfusionMatrix EvaluateClassifier(KStrongClassifierCoord& classifier,
                                   const std::vector<KVector>& lx,
                                   const KVector& vY)
{
    KVector vYp = classifier(lx);

    ConfusionMatrix cm = {0, 0, 0, 0, 0.0, 0.0};
    int nNum = lx.size();

    for (int i = 0; i < nNum; i++) {
        int pred   = (int)vYp[i];
        int actual = (int)vY[i];
        if (actual == 1 && pred == 1)        cm.TP++;
        else if (actual == -1 && pred == 1)  cm.FP++;
        else if (actual == 1 && pred == -1)  cm.FN++;
        else                                 cm.TN++;
    }

    cm.error    = (double)(cm.FP + cm.FN) / (double)nNum;
    cm.accuracy = (double)(cm.TP + cm.TN) / (double)nNum * 100.0;

    return cm;
}
