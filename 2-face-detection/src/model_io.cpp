#include "model_io.h"
#include <QFile>
#include <QTextStream>
#include <iostream>

bool saveCascadeModel(const CascadeClassifier& cascade, const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return false;
    QTextStream out(&file);

    const auto& stages = cascade.getStages();

    out << "CASCADE_MODEL_V1\n";
    out << "LAYERS " << stages.size();
    for (size_t s = 0; s < stages.size(); s++)
        out << " " << stages[s].classifiers.size();
    out << "\n";
    out << "STAGES " << stages.size() << "\n";

    for (size_t s = 0; s < stages.size(); s++) {
        const auto& stage = stages[s];
        out << "STAGE " << s << " " << stage.classifiers.size()
            << " " << stage.detectThreshold << "\n";
        for (size_t w = 0; w < stage.classifiers.size(); w++) {
            const auto& wc = stage.classifiers[w];
            out << "WC " << wc.featureIndex << " " << wc.threshold
                << " " << wc.polarity << " " << wc.alpha << "\n";
        }
    }

    file.close();
    std::cout << "Model saved to: " << path.toStdString() << std::endl;
    return true;
}

CascadeClassifier* loadCascadeModel(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        std::cout << "Cannot open model: " << path.toStdString() << std::endl;
        return nullptr;
    }
    QTextStream in(&file);

    QString header = in.readLine().trimmed();
    if (header != "CASCADE_MODEL_V1") {
        std::cout << "Invalid model format" << std::endl;
        return nullptr;
    }

    // Read layers
    QString layersLine = in.readLine().trimmed();
    QStringList layerParts = layersLine.split(' ', Qt::SkipEmptyParts);
    std::vector<int> layers;
    for (int i = 1; i < layerParts.size(); i++)
        layers.push_back(layerParts[i].toInt());

    // Read number of stages
    QString stagesLine = in.readLine().trimmed();
    int numStages = stagesLine.split(' ', Qt::SkipEmptyParts)[1].toInt();

    // Generate all Haar features from a dummy 24x24 image
    KImageGray dummy(24, 24);
    dummy.Cleared();
    AdaBoost featureGen(1);
    auto allFeatures = featureGen.extractAllHaarFeatures(dummy);
    std::cout << "Model load: regenerated " << allFeatures.size() << " features" << std::endl;

    CascadeClassifier* cascade = new CascadeClassifier(layers);

    for (int s = 0; s < numStages; s++) {
        QString stageLine = in.readLine().trimmed();
        QStringList stageParts = stageLine.split(' ', Qt::SkipEmptyParts);
        int numWC = stageParts[2].toInt();
        double detectThresh = stageParts[3].toDouble();

        AdaBoost stage(numWC);
        stage.detectThreshold = detectThresh;
        stage.allFeatures = allFeatures;

        for (int w = 0; w < numWC; w++) {
            QString wcLine = in.readLine().trimmed();
            QStringList wcParts = wcLine.split(' ', Qt::SkipEmptyParts);
            int featIdx = wcParts[1].toInt();
            double thresh = wcParts[2].toDouble();
            int polarity = wcParts[3].toInt();
            double alpha = wcParts[4].toDouble();

            WeakClassifier wc;
            wc.featureIndex = featIdx;
            wc.threshold = thresh;
            wc.polarity = polarity;
            wc.alpha = alpha;
            wc.feature = allFeatures[featIdx].get();
            stage.classifiers.push_back(wc);
        }

        cascade->addStage(stage);
    }

    file.close();
    std::cout << "Model loaded: " << numStages << " stages from " << path.toStdString() << std::endl;
    return cascade;
}
