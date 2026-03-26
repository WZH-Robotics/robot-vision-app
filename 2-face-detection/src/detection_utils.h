#ifndef DETECTION_UTILS_H
#define DETECTION_UTILS_H

#include <vector>
#include "adaboost.h"
#include "cascade_classifier.h"
#include "kfc.h"

// IoU (Intersection over Union) between two detection boxes
double computeIoU(const DetectionResult& a, const DetectionResult& b);

// IoU between two QRects
double computeQRectIoU(const QRect& a, const QRect& b);

// NMS: group overlapping detections, keep groups with >= minNeighbors
std::vector<DetectionResult> groupDetections(std::vector<DetectionResult>& detections, double iouThreshold, int minNeighbors);

// Draw a red bounding box on a color image
void drawRedBox(KImageColor& colorImage, const DetectionResult& result);

// Multi-scale sliding window detection
std::vector<DetectionResult> detectMultiScale(const KImageGray& img, CascadeClassifier* cascade,
                                              int winSize, int step, double scaleFactor);

#endif // DETECTION_UTILS_H
