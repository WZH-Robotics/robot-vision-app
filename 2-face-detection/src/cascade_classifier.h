#ifndef CASCADE_CLASSIFIER_H
#define CASCADE_CLASSIFIER_H

#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <QString>
#include <QImage>
#include <QColor>
#include <QRect>
#include "adaboost.h"

struct FDDBImageInfo {
    QString imagePath;
    std::vector<QRect> faceBboxes;
};

// IoU between two QRects (for hard negative mining)
static double computeQRectIoU_cascade(const QRect& a, const QRect& b) {
    QRect inter = a.intersected(b);
    if (inter.isEmpty()) return 0.0;
    int interArea = inter.width() * inter.height();
    int unionArea = a.width() * a.height() + b.width() * b.height() - interArea;
    if (unionArea <= 0) return 0.0;
    return (double)interArea / unionArea;
}

class CascadeClassifier {
public:
    CascadeClassifier() = default;
    CascadeClassifier(const std::vector<int>& layers) : layers(layers) {}
    std::vector<DetectionResult> detectionResults;

    // ===== Paper-based Viola-Jones cascade training =====
    void train(const std::vector<std::pair<KImageGray, int>>& training,
               const std::vector<FDDBImageInfo>& fddbInfos,
               int maxNegPerStage = 1500) {

        std::cout << "=== Paper-based Viola-Jones Cascade Training ===" << std::endl;

        // Parameters (Viola-Jones 2001)
        const double d_min = 0.95;      // minimum detection rate per stage (lowered for generalization)
        const double f_max = 0.5;       // maximum FPR per stage (not used as hard constraint, just reference)
        const double F_target = 1e-6;   // overall cascade FPR target
        const int max_stages = 6;
        const std::vector<int> fixedT = {2, 10, 25, 50, 50, 50};

        // Separate positives from training data
        std::vector<std::pair<KImageGray, int>> pos;
        for (const auto& ex : training) {
            if (ex.second == 1) pos.push_back(ex);
        }

        // Initial negatives from training data
        std::vector<std::pair<KImageGray, int>> neg;
        for (const auto& ex : training) {
            if (ex.second == 0) neg.push_back(ex);
        }

        double overallFPR = 1.0;

        for (int stageIdx = 0; stageIdx < max_stages; stageIdx++) {

            // --- Hard Negative Mining (from stage 1 onwards) ---
            if (stageIdx > 0 && !fddbInfos.empty()) {
                std::cout << "\n--- Hard Negative Mining for Stage " << stageIdx << " ---" << std::endl;
                std::vector<std::pair<KImageGray, int>> hardNegs;
                mineHardNegatives(fddbInfos, hardNegs, maxNegPerStage);

                if ((int)hardNegs.size() < 10) {
                    std::cout << "  Only " << hardNegs.size() << " hard negatives found (< 10). Cascade FPR is very low. Stopping." << std::endl;
                    break;
                }
                neg = hardNegs;
            }

            if (neg.empty()) {
                std::cout << "Stage " << stageIdx << ": no negatives available. Stopping." << std::endl;
                break;
            }

            // Determine T for this stage
            int T = (stageIdx < (int)fixedT.size()) ? fixedT[stageIdx] : fixedT.back();

            // Build combined training set
            std::vector<std::pair<KImageGray, int>> combined = pos;
            combined.insert(combined.end(), neg.begin(), neg.end());

            std::cout << "\n=== Cascade Stage " << stageIdx << " (T=" << T
                      << ", pos=" << pos.size() << ", neg=" << neg.size() << ") ===" << std::endl;

            // Train AdaBoost with T weak classifiers
            AdaBoost clf(T);
            clf.train(combined, pos.size(), neg.size());

            // --- Threshold Adjustment: guarantee d_min detection rate ---
            // Compute confidence scores on positives
            std::vector<double> posScores;
            for (const auto& ex : pos) {
                double score = clf.confidenceScore(ex.first);
                posScores.push_back(score);
            }
            std::sort(posScores.begin(), posScores.end());

            // Find threshold that keeps at least d_min fraction of positives
            // d_min = 0.995 means we allow 0.5% of positives to be missed
            int idx = (int)((1.0 - d_min) * posScores.size());
            if (idx < 0) idx = 0;
            if (idx >= (int)posScores.size()) idx = (int)posScores.size() - 1;
            double stageThreshold = posScores[idx];

            clf.detectThreshold = stageThreshold;

            // Measure actual detection rate and FPR for this stage
            int posCorrect = 0;
            for (const auto& ex : pos) {
                if (clf.classify(ex.first) == 1) posCorrect++;
            }
            double stageDR = (double)posCorrect / pos.size();

            int fpCount = 0;
            for (const auto& ex : neg) {
                if (clf.classify(ex.first) == 1) fpCount++;
            }
            double stageFPR = (double)fpCount / neg.size();

            overallFPR *= stageFPR;

            std::cout << "  Stage " << stageIdx << " threshold=" << stageThreshold
                      << " DR=" << stageDR << " FPR=" << stageFPR << std::endl;
            std::cout << "  Overall cascade FPR=" << overallFPR << std::endl;

            // Store this stage
            clfs.push_back(clf);
            layers.push_back(T);

            // Check termination: overall FPR below target
            if (overallFPR < F_target) {
                std::cout << "\n*** Target FPR " << F_target << " reached (current: "
                          << overallFPR << "). Stopping cascade. ***" << std::endl;
                break;
            }
        }

        std::cout << "\n=== Cascade training complete: " << clfs.size() << " stages ===" << std::endl;
    }

    void setDetectThreshold(double threshold) {
        for (auto& clf : clfs) {
            clf.detectThreshold = threshold;
        }
    }

    int classify(const KImageGray& image) {
        for (const auto& clf : clfs) {
            int result = clf.classify(image);
            if (result == 0) return 0;
        }
        return 1;
    }

    void saveInfo(const std::string& filename) const {
        std::ofstream txt(filename);
        if (!txt) return;
        txt << "=== Cascade Classifier Info ===" << std::endl;
        txt << "Total stages: " << clfs.size() << std::endl << std::endl;
        for (size_t s = 0; s < clfs.size(); s++) {
            txt << "--- Stage " << s << " (weak classifiers: " << clfs[s].classifiers.size() << ") ---" << std::endl;
            for (size_t w = 0; w < clfs[s].classifiers.size(); w++) {
                const auto& wc = clfs[s].classifiers[w];
                txt << "  WC[" << w << "]: FeatureIdx=" << wc.featureIndex
                    << ", dThresh=" << wc.threshold
                    << ", polarity=" << wc.polarity
                    << ", alpha=" << wc.alpha << std::endl;
            }
            txt << std::endl;
        }
        txt.close();
        std::cout << "Classifier info saved to: " << filename << std::endl;
    }

    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) return;
        size_t numClfs = clfs.size();
        file.write(reinterpret_cast<const char*>(&numClfs), sizeof(numClfs));
        for (const auto& clf : clfs) {
            (void)clf;
        }
    }

    static CascadeClassifier load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return CascadeClassifier();
        CascadeClassifier cascade;
        size_t numClfs;
        file.read(reinterpret_cast<char*>(&numClfs), sizeof(numClfs));
        return cascade;
    }

    const std::vector<AdaBoost>& getStages() const { return clfs; }
    const std::vector<int>& getLayers() const { return layers; }
    void addStage(const AdaBoost& stage) { clfs.push_back(stage); }

private:
    std::vector<AdaBoost> classifiers;
    std::vector<int> layers;
    std::vector<AdaBoost> clfs;

    // Hard Negative Mining: extract patches from FDDB images,
    // classify through current cascade, collect false positives
    void mineHardNegatives(const std::vector<FDDBImageInfo>& fddbInfos,
                           std::vector<std::pair<KImageGray, int>>& hardNegs,
                           int maxCount) {
        std::mt19937 rng(42);
        const int patchesPerImage = 50;
        const int patchSize = 24;
        int collected = 0;

        std::cout << "  Mining from " << fddbInfos.size() << " FDDB images..." << std::endl;

        for (size_t imgIdx = 0; imgIdx < fddbInfos.size() && collected < maxCount; imgIdx++) {
            QImage qImg(fddbInfos[imgIdx].imagePath);
            if (qImg.isNull()) continue;

            int imgW = qImg.width();
            int imgH = qImg.height();
            if (imgW <= patchSize || imgH <= patchSize) continue;

            std::uniform_int_distribution<int> distX(0, imgW - patchSize - 1);
            std::uniform_int_distribution<int> distY(0, imgH - patchSize - 1);

            for (int p = 0; p < patchesPerImage && collected < maxCount; p++) {
                int px = distX(rng);
                int py = distY(rng);
                QRect patchRect(px, py, patchSize, patchSize);

                // Skip patches overlapping with face bboxes
                bool overlaps = false;
                for (const QRect& bbox : fddbInfos[imgIdx].faceBboxes) {
                    if (computeQRectIoU_cascade(patchRect, bbox) > 0.1) {
                        overlaps = true;
                        break;
                    }
                }
                if (overlaps) continue;

                // Extract 24x24 grayscale patch
                QImage patch = qImg.copy(px, py, patchSize, patchSize);
                KImageGray grayPatch(patchSize, patchSize);
                for (int y = 0; y < patchSize; y++) {
                    for (int x = 0; x < patchSize; x++) {
                        QColor c = patch.pixelColor(x, y);
                        grayPatch[y][x] = static_cast<unsigned char>(
                            0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue());
                    }
                }

                // Classify through current cascade: if it passes (false positive), it's a hard negative
                if (this->classify(grayPatch) == 1) {
                    hardNegs.push_back({grayPatch, 0});
                    collected++;
                }
            }

            // Progress log every 500 images
            if (imgIdx % 500 == 0 && imgIdx > 0) {
                std::cout << "    Processed " << imgIdx << "/" << fddbInfos.size()
                          << " images, collected " << collected << " hard negatives" << std::endl;
            }
        }

        std::cout << "  Hard negative mining: collected " << collected << " FPs" << std::endl;
    }
};

#endif // CASCADE_CLASSIFIER_H
