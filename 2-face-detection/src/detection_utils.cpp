#include "detection_utils.h"
#include <algorithm>
#include <cmath>

double computeIoU(const DetectionResult& a, const DetectionResult& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    int intersection = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.width * a.height + b.width * b.height - intersection;
    if (unionArea <= 0) return 0.0;
    return (double)intersection / unionArea;
}

double computeQRectIoU(const QRect& a, const QRect& b) {
    QRect inter = a.intersected(b);
    if (inter.isEmpty()) return 0.0;
    int interArea = inter.width() * inter.height();
    int unionArea = a.width() * a.height() + b.width() * b.height() - interArea;
    if (unionArea <= 0) return 0.0;
    return (double)interArea / unionArea;
}

std::vector<DetectionResult> groupDetections(std::vector<DetectionResult>& detections, double iouThreshold, int minNeighbors) {
    if (detections.empty()) return {};

    std::vector<bool> merged(detections.size(), false);
    std::vector<DetectionResult> result;

    for (size_t i = 0; i < detections.size(); i++) {
        if (merged[i]) continue;

        std::vector<size_t> group;
        group.push_back(i);
        merged[i] = true;

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (merged[j]) continue;
            for (size_t ki = 0; ki < group.size(); ki++) {
                if (computeIoU(detections[j], detections[group[ki]]) > iouThreshold) {
                    group.push_back(j);
                    merged[j] = true;
                    break;
                }
            }
        }

        if ((int)group.size() >= minNeighbors) {
            double avgX = 0, avgY = 0, avgW = 0, avgH = 0;
            for (size_t idx : group) {
                avgX += detections[idx].x;
                avgY += detections[idx].y;
                avgW += detections[idx].width;
                avgH += detections[idx].height;
            }
            int n = group.size();
            result.push_back(DetectionResult((int)(avgX/n), (int)(avgY/n), (int)(avgW/n), (int)(avgH/n)));
        }
    }
    return result;
}

void drawRedBox(KImageColor& colorImage, const DetectionResult& result) {
    int rows = colorImage.Row(), cols = colorImage.Col();
    // Top and bottom edges
    for (int x = std::max(0, result.x); x < std::min(cols, result.x + result.width); ++x) {
        if (result.y >= 0 && result.y < rows) {
            colorImage[result.y][x].r = 255;
            colorImage[result.y][x].g = 0;
            colorImage[result.y][x].b = 0;
        }
        int bottomY = result.y + result.height - 1;
        if (bottomY >= 0 && bottomY < rows) {
            colorImage[bottomY][x].r = 255;
            colorImage[bottomY][x].g = 0;
            colorImage[bottomY][x].b = 0;
        }
    }
    // Left and right edges
    for (int y = std::max(0, result.y); y < std::min(rows, result.y + result.height); ++y) {
        if (result.x >= 0 && result.x < cols) {
            colorImage[y][result.x].r = 255;
            colorImage[y][result.x].g = 0;
            colorImage[y][result.x].b = 0;
        }
        int rightX = result.x + result.width - 1;
        if (rightX >= 0 && rightX < cols) {
            colorImage[y][rightX].r = 255;
            colorImage[y][rightX].g = 0;
            colorImage[y][rightX].b = 0;
        }
    }
}

std::vector<DetectionResult> detectMultiScale(const KImageGray& img, CascadeClassifier* cascade,
                                              int winSize, int step, double scaleFactor) {
    std::vector<DetectionResult> detectionResults;
    int origW = img.Col();
    int origH = img.Row();
    double scale = 1.0;

    while (true) {
        int scaledW = static_cast<int>(origW / scale);
        int scaledH = static_cast<int>(origH / scale);
        if (scaledW < winSize || scaledH < winSize) break;

        KImageGray scaledImg(scaledH, scaledW);
        for (int sy = 0; sy < scaledH; sy++) {
            for (int sx = 0; sx < scaledW; sx++) {
                double srcXf = sx * scale;
                double srcYf = sy * scale;
                int x0 = static_cast<int>(srcXf);
                int y0 = static_cast<int>(srcYf);
                int x1 = std::min(x0 + 1, origW - 1);
                int y1 = std::min(y0 + 1, origH - 1);
                x0 = std::min(x0, origW - 1);
                y0 = std::min(y0, origH - 1);
                double dx = srcXf - x0;
                double dy = srcYf - y0;
                double val = (1-dx)*(1-dy)*img[y0][x0] + dx*(1-dy)*img[y0][x1]
                           + (1-dx)*dy*img[y1][x0] + dx*dy*img[y1][x1];
                scaledImg[sy][sx] = static_cast<unsigned char>(val);
            }
        }

        for (int wy = 0; wy <= scaledH - winSize; wy += step) {
            for (int wx = 0; wx <= scaledW - winSize; wx += step) {
                KImageGray window(winSize, winSize);
                for (int j = 0; j < winSize; j++)
                    for (int i = 0; i < winSize; i++)
                        window[j][i] = scaledImg[wy + j][wx + i];

                if (cascade->classify(window) == 1) {
                    int origX = static_cast<int>(wx * scale);
                    int origY = static_cast<int>(wy * scale);
                    int origSize = static_cast<int>(winSize * scale);
                    detectionResults.push_back(DetectionResult(origX, origY, origSize, origSize));
                }
            }
        }
        scale *= scaleFactor;
    }

    return detectionResults;
}
