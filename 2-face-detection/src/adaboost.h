#ifndef ADABOOST_H
#define ADABOOST_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include "kfc.h"

struct DetectionResult {
    int x, y, width, height;
    DetectionResult(int x, int y, int width, int height) : x(x), y(y), width(width), height(height) {}
};

// 약 분류기 구현
class WeakClassifier {
public:
    KFeatureHaarlike* feature;
    double threshold;
    int polarity;
    double alpha;
    int featureIndex;

    // Default constructor
    WeakClassifier() : feature(nullptr), threshold(0.0), polarity(0), alpha(0.0), featureIndex(-1) {}

    // Constructor with parameters
    WeakClassifier(KFeatureHaarlike* feature, double threshold, int polarity, int featureIndex = -1)
        : feature(feature), threshold(threshold), polarity(polarity), featureIndex(featureIndex) {}

    int classify(const KImageDouble& img) const {
        if (!feature) return 0; // Safe guard in case feature is nullptr
        double featureValue = (*feature)(img);
        return (polarity * featureValue < polarity * threshold) ? 1 : 0;
    }

    int classify(const KImageDouble& img, double dMean, double dStd) const {
        if (!feature) return 0;
        double featureValue = (*feature)(img, dMean, dStd);
        return (polarity * featureValue < polarity * threshold) ? 1 : 0;
    }
};


// AdaBoost 알고리즘 클래스
class AdaBoost {
public:
    std::vector<WeakClassifier> classifiers;
    std::vector<std::shared_ptr<KFeatureHaarlike>> allFeatures;
    std::vector<std::vector<double>> lastFeatures; // cached after train()
    std::vector<double> trainMeans, trainStds; // per-image normalization stats
    int T;
    double detectThreshold;

    // Compute per-window mean and stddev for variance normalization
    static void computeWindowStats(const KImageGray& img, double& dMean, double& dStd) {
        double sum = 0, sqSum = 0;
        int total = img.Row() * img.Col();
        for (int y = 0; y < img.Row(); y++)
            for (int x = 0; x < img.Col(); x++) {
                double v = img[y][x];
                sum += v;
                sqSum += v * v;
            }
        dMean = sum / total;
        double var = sqSum / total - dMean * dMean;
        dStd = (var > 1.0) ? sqrt(var) : 1.0;
    }

    AdaBoost(int T) : T(T), detectThreshold(0.5) {}

    std::vector<std::shared_ptr<KFeatureHaarlike>> extractAllHaarFeatures(const KImageGray& grayImg) {
        KImageDouble idInt = toIntegralImage(grayImg);
        int width = grayImg.Col();
        int height = grayImg.Row();
        std::vector<std::shared_ptr<KFeatureHaarlike>> features;

        double sum = 0, sqSum = 0;
        int totalPixels = width * height;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                sum += grayImg[y][x];
                sqSum += grayImg[y][x] * grayImg[y][x];
            }
        }
        double mean = sum / totalPixels;
        double variance = (sqSum / totalPixels) - (mean * mean);
        double stdDev = sqrt(variance);

        for (int w = 2; w <= width; w += 2) {
            for (int h = 1; h <= height; ++h) {
                for (int x = 0; x <= width - w; ++x) {
                    for (int y = 0; y <= height - h; ++y) {
                        features.emplace_back(std::make_shared<KFeatureHaarlike2H>(x, y, w, h));
                    }
                }
            }
        }

        for (int h = 2; h <= height; h += 2) {
            for (int w = 1; w <= width; ++w) {
                for (int y = 0; y <= height - h; ++y) {
                    for (int x = 0; x <= width - w; ++x) {
                        features.emplace_back(std::make_shared<KFeatureHaarlike2V>(x, y, w, h));
                    }
                }
            }
        }

        for (int w = 3; w <= width; w += 3) {
            for (int h = 1; h <= height; ++h) {
                for (int x = 0; x <= width - w; ++x) {
                    for (int y = 0; y <= height - h; ++y) {
                        features.emplace_back(std::make_shared<KFeatureHaarlike3H>(x, y, w, h));
                    }
                }
            }
        }

        for (int h = 3; h <= height; h += 3) {
            for (int w = 1; w <= width; ++w) {
                for (int y = 0; y <= height - h; ++y) {
                    for (int x = 0; x <= width - w; ++x) {
                        features.emplace_back(std::make_shared<KFeatureHaarlike3V>(x, y, w, h));
                    }
                }
            }
        }

        for (int w = 2; w <= width; w += 2) {
            for (int h = 2; h <= height; h += 2) {
                for (int x = 0; x <= width - w; ++x) {
                    for (int y = 0; y <= height - h; ++y) {
                        features.emplace_back(std::make_shared<KFeatureHaarlike4>(x, y, w, h));
                    }
                }
            }
        }

        //std::cout << "Total number of 2H features: " << features.size() << std::endl;

        if (allFeatures.empty()) {
            allFeatures = features;
        }

        return features;
    }


    std::vector<std::vector<double>> extractFeaturesFromImages(const std::vector<std::pair<KImageGray, int>>& training_data, std::vector<int>& y) {
        std::vector<std::vector<double>> X;
        for (size_t i = 0; i < training_data.size(); ++i) {
            const KImageGray& img = training_data[i].first;
            KImageDouble integralImage = toIntegralImage(img);
            std::vector<std::shared_ptr<KFeatureHaarlike>> features = extractAllHaarFeatures(img);
            std::vector<double> featureValues;
            featureValues.reserve(features.size());

            double dMean = (i < trainMeans.size()) ? trainMeans[i] : 0.0;
            double dStd = (i < trainStds.size()) ? trainStds[i] : 1.0;

            for (const auto& feature : features) {
                featureValues.push_back((*feature)(integralImage, dMean, dStd));
            }

            X.push_back(std::move(featureValues));
            y.push_back(training_data[i].second);

        }


        return X;
    }


    // Pre-compute integral images, labels, weights from training data
    void prepareData(std::vector<std::pair<KImageGray, int>>& training_data, int pos_num, int neg_num,
                     std::vector<KImageDouble>& integralImages,
                     std::vector<std::pair<KImageDouble, int>>& trainingDataDouble,
                     std::vector<double>& weights, std::vector<int>& y) {
        integralImages.resize(training_data.size());
        trainingDataDouble.resize(training_data.size());
        weights.resize(training_data.size());
        trainMeans.resize(training_data.size());
        trainStds.resize(training_data.size());
        for (size_t i = 0; i < training_data.size(); ++i) {
            integralImages[i] = toIntegralImage(training_data[i].first);
            trainingDataDouble[i] = {KImageDouble(training_data[i].first), training_data[i].second};
            weights[i] = (training_data[i].second == 1) ? 1.0 / (2 * pos_num) : 1.0 / (2 * neg_num);
            computeWindowStats(training_data[i].first, trainMeans[i], trainStds[i]);
        }
    }

    // Train with pre-computed feature matrix (fast: skips feature extraction)
    void trainWithFeatures(const std::vector<std::vector<double>>& features, const std::vector<int>& y,
                           std::vector<double>& weights,
                           std::vector<std::pair<KImageDouble, int>>& trainingDataDouble,
                           std::vector<KImageDouble>& integralImages,
                           std::vector<std::pair<KImageGray, int>>& training_data) {
        for (int t = 0; t < T; ++t) {
            normalizeWeights(weights);

            auto weakClassifiers = train_weak(features, y, weights);
            auto best = selectBestClassifier(weakClassifiers, weights, trainingDataDouble, integralImages);
            double err = std::max(best.second, 1e-10); // avoid division by zero
            err = std::min(err, 1.0 - 1e-10);
            double beta = err / (1.0 - err);
            double alpha = log(1.0 / beta);

            for (size_t i = 0; i < training_data.size(); ++i) {
                double dMean = (i < trainMeans.size()) ? trainMeans[i] : 0.0;
                double dStd = (i < trainStds.size()) ? trainStds[i] : 1.0;
                int prediction = best.first.classify(integralImages[i], dMean, dStd);
                if (prediction == training_data[i].second) {
                    weights[i] *= beta;
                }
            }

            best.first.alpha = alpha;
            this->classifiers.push_back(best.first);
            std::cout << "  Round " << t << ": featureIdx=" << best.first.featureIndex
                      << " thresh=" << best.first.threshold
                      << " alpha=" << alpha << std::endl;
        }
    }

    // Original train (extracts features itself)
    void train(std::vector<std::pair<KImageGray, int>>& training_data, int pos_num, int neg_num) {
        std::vector<KImageDouble> integralImages;
        std::vector<std::pair<KImageDouble, int>> trainingDataDouble;
        std::vector<double> weights;
        std::vector<int> y;

        prepareData(training_data, pos_num, neg_num, integralImages, trainingDataDouble, weights, y);
        std::cout << "weight initialize done" << std::endl;

        std::vector<std::vector<double>> features = extractFeaturesFromImages(training_data, y);
        std::cout << "extract feature done" << std::endl;

        trainWithFeatures(features, y, weights, trainingDataDouble, integralImages, training_data);
        lastFeatures = std::move(features); // cache for cascade reuse
    }



    std::vector<WeakClassifier> train_weak(const std::vector<std::vector<double>>& X, const std::vector<int>& y, const std::vector<double>& weights) {
        if (X.empty() || y.empty() || weights.empty()) {
            std::cerr << "Input vectors are empty." << std::endl;
            return {};
        }

        std::vector<WeakClassifier> classifiers;
        double totalPos = 0.0, totalNeg = 0.0;
        int dataSize = y.size();
        int featureCount = X[0].size();

        // Check all rows have same feature count
        for (int i = 0; i < dataSize; ++i) {
            if ((int)X[i].size() != featureCount) {
                std::cerr << "ERROR: X[" << i << "].size()=" << X[i].size() << " != " << featureCount << std::endl;
            }
        }

        classifiers.reserve(featureCount);  // pre-allocate to avoid reallocation crashes

        std::cout << "train_weak: dataSize=" << dataSize << " featureCount=" << featureCount << std::endl;

        for (int i = 0; i < dataSize; ++i) {
            if (std::isnan(weights[i]) || std::isinf(weights[i])) {
                std::cerr << "Invalid weight value detected at " << i << std::endl;
                return {};
            }
            if (y[i] == 1) totalPos += weights[i];
            else totalNeg += weights[i];
        }

        if ((int)allFeatures.size() < featureCount) {
            std::cerr << "Feature count mismatch. Expected at least " << featureCount << ", got " << allFeatures.size() << std::endl;
            return classifiers;
        }

        for (int featureIndex = 0; featureIndex < featureCount; ++featureIndex) {
            if (!allFeatures[featureIndex]) {
                std::cerr << "Null feature pointer at index: " << featureIndex << std::endl;
                continue;
            }

            // NaN check & build sorted list
            std::vector<std::tuple<double, double, int>> featureWithWeight;
            bool hasNaN = false;
            for (int i = 0; i < dataSize; ++i) {
                double val = X[i][featureIndex];
                if (std::isnan(val) || std::isinf(val)) {
                    hasNaN = true;
                    break;
                }
                featureWithWeight.emplace_back(weights[i], val, y[i]);
            }
            if (hasNaN) {
                // skip this feature - NaN would crash std::sort
                classifiers.emplace_back(allFeatures[featureIndex].get(), 0.0, 1);
                continue;
            }

            std::sort(featureWithWeight.begin(), featureWithWeight.end(), [](const auto& a, const auto& b) {
                return std::get<1>(a) < std::get<1>(b);
            });

            double posSeen = 0.0, negSeen = 0.0, posWeights = 0.0, negWeights = 0.0;
            double minError = std::numeric_limits<double>::infinity();
            double bestThreshold = 0.0;
            int bestPolarity = 1;

            for (const auto& data : featureWithWeight) {
                double w = std::get<0>(data);
                double f = std::get<1>(data);
                int label = std::get<2>(data);

                if (label == 1) {
                    posSeen += 1;
                    posWeights += w;
                } else {
                    negSeen += 1;
                    negWeights += w;
                }

                double e1 = negWeights + (totalPos - posWeights);
                double e2 = posWeights + (totalNeg - negWeights);
                double error = std::min(e1, e2);
                if (error < minError) {
                    minError = error;
                    bestThreshold = f;
                    bestPolarity = (e1 < e2) ? 1 : -1;
                }
            }

            if (minError == std::numeric_limits<double>::infinity()) {
                std::cerr << "Error calculation failed; no valid error found." << std::endl;
            }

            KFeatureHaarlike* featurePtr = allFeatures[featureIndex].get();
            classifiers.emplace_back(featurePtr, bestThreshold, bestPolarity, featureIndex);
        }

        return classifiers;
    }




    /*
    std::vector<std::pair<std::vector<KFeatureHaarlike*>, std::vector<KFeatureHaarlike*>>> buildFeatures(int width, int height) {
        std::vector<std::pair<std::vector<KFeatureHaarlike*>, std::vector<KFeatureHaarlike*>>> feature_pairs;

        // 2H, 2V, 3H, 3V 및 4-type 특징 생성
        for (int w = 1; w <= width; w++) {
            for (int h = 1; h <= height; h++) {
                for (int x = 0; x <= width - w; x++) {
                    for (int y = 0; y <= height - h; y++) {
                        std::vector<KFeatureHaarlike*> positive_features;
                        std::vector<KFeatureHaarlike*> negative_features;

                        // 예시로 모든 특징을 긍정적 특징으로 처리
                        if (w % 2 == 0) {  // 가로로 나눌 수 있는 경우
                            positive_features.push_back(new KFeatureHaarlike2H(x, y, w, h));
                        }
                        if (h % 2 == 0) {  // 세로로 나눌 수 있는 경우
                            positive_features.push_back(new KFeatureHaarlike2V(x, y, w, h));
                        }
                        if (w % 3 == 0) {  // 가로로 3등분 가능한 경우
                            positive_features.push_back(new KFeatureHaarlike3H(x, y, w, h));
                        }
                        if (h % 3 == 0) {  // 세로로 3등분 가능한 경우
                            positive_features.push_back(new KFeatureHaarlike3V(x, y, w, h));
                        }
                        if (w % 2 == 0 && h % 2 == 0) {  // 가로 세로 모두 2로 나눌 수 있는 경우
                            positive_features.push_back(new KFeatureHaarlike4(x, y, w, h));
                        }

                        // 현재 긍정적 특징과 부정적 특징을 하나의 쌍으로 묶어 저장
                        feature_pairs.push_back(std::make_pair(positive_features, negative_features));
                    }
                }
            }
        }

        return feature_pairs;
    }
    */
    void normalizeWeights(std::vector<double>& weights) {
        double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        std::transform(weights.begin(), weights.end(), weights.begin(), [sum](double w) { return w / sum; });
    }

    std::pair<WeakClassifier, double> selectBestClassifier(
        const std::vector<WeakClassifier>& classifiers,
        const std::vector<double>& weights,
        const std::vector<std::pair<KImageDouble, int>>& training_data,
        const std::vector<KImageDouble>& integralImages) {

        WeakClassifier bestClassifier;
        double bestError = 1;   // = std::numeric_limits<double>::infinity();
        std::vector<int> bestAccuracy(training_data.size(), 0);

        for (const auto& classifier : classifiers) {
            double error = 0.0;
            std::vector<int> accuracy(training_data.size(), 0);

            for (size_t i = 0; i < training_data.size(); ++i) {
                const auto& [image, label] = training_data[i];
                double dMean = (i < trainMeans.size()) ? trainMeans[i] : 0.0;
                double dStd = (i < trainStds.size()) ? trainStds[i] : 1.0;
                int prediction = classifier.classify(integralImages[i], dMean, dStd);
                bool isCorrect = prediction == label;
                accuracy[i] = isCorrect ? 0 : 1;
                error += weights[i] * accuracy[i];
            }

            if (error < bestError) {
                bestError = error;
                bestClassifier = classifier;
                bestAccuracy = accuracy;
            }
        }

        // Output the best error for debugging purposes
        std::cout << "Best error: " << bestError << std::endl;

        return {bestClassifier, bestError};  // Optionally add 'bestAccuracy' to the return value if needed
    }

    /*
    void applyFeatures(
        std::vector<std::pair<std::vector<KFeatureHaarlike*>, std::vector<KFeatureHaarlike*>>>& features,
        const std::vector<KImageDouble>& integralImages,
        const std::vector<int>& labels,
        std::vector<std::vector<double>>& X,
        std::vector<int>& y)
    {
        size_t num_features = features.size();
        size_t num_samples = integralImages.size();
        if (num_features == 0 || num_samples == 0) {
            std::cerr << "Error: Empty features or integralImages vectors." << std::endl;
            return;
        }
        if (num_features != features[0].first.size() || num_features != features[0].second.size()) {
            std::cout<<features[0].first.size()<<std::endl;
            std::cerr << "Error: Inconsistent features vector sizes." << std::endl;
            return;
        }
        if (num_samples != labels.size()) {
            std::cerr << "Error: Size mismatch between integralImages and labels vectors." << std::endl;
            return;
        }

        X.resize(num_features, std::vector<double>(num_samples));
        y.resize(num_samples);

        for (size_t i = 0; i < num_samples; ++i) {
            y[i] = labels[i];  // Copy labels from the provided vector
            for (size_t j = 0; j < num_features; ++j) {
                double featureValue = 0;
                for (auto& pos_feature : features[j].first) {
                    featureValue += (*pos_feature)(integralImages[i]);
                }
                for (auto& neg_feature : features[j].second) {
                    featureValue -= (*neg_feature)(integralImages[i]);
                }
                X[j][i] = featureValue;
            }
        }
    }
    */

    double confidenceScore(const KImageGray& image) const {
        double dMean, dStd;
        computeWindowStats(image, dMean, dStd);
        KImageDouble ii = toIntegralImage(image);
        double total = 0;
        for (const auto& clf : classifiers)
            total += clf.alpha * clf.classify(ii, dMean, dStd);
        return total;
    }

    double alphaSum() const {
        return std::accumulate(classifiers.begin(), classifiers.end(), 0.0,
            [](double sum, const WeakClassifier& clf) { return sum + clf.alpha; });
    }

    // classify using absolute threshold (set by cascade training)
    int classify(const KImageGray& image) const {
        return confidenceScore(image) >= detectThreshold ? 1 : 0;
    }

    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for writing.");

        size_t size = classifiers.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        for (const auto& clf : classifiers) {
            double threshold = clf.threshold;
            double alpha = clf.alpha;
            int polarity = clf.polarity;
            file.write(reinterpret_cast<const char*>(&threshold), sizeof(threshold));
            file.write(reinterpret_cast<const char*>(&polarity), sizeof(polarity));
            file.write(reinterpret_cast<const char*>(&alpha), sizeof(alpha));
            // Note: Saving feature pointers directly is not practical; feature definitions or IDs should be saved and reconstructed.
        }
        file.close();
    }


    static AdaBoost load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for reading.");

        AdaBoost loadedBoost(0);
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        loadedBoost.classifiers.resize(size);
        for (auto& clf : loadedBoost.classifiers) {
            file.read(reinterpret_cast<char*>(&clf.threshold), sizeof(clf.threshold));
            file.read(reinterpret_cast<char*>(&clf.polarity), sizeof(clf.polarity));
            file.read(reinterpret_cast<char*>(&clf.alpha), sizeof(clf.alpha));
            // Note: Reconstructing features from IDs or definitions is necessary.
        }
        file.close();
        return loadedBoost;
    }
};


#endif // ADABOOST_H
