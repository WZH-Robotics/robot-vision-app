#include "fddb_utils.h"
#include <QFile>
#include <QTextStream>
#include <QColor>
#include <iostream>
#include <algorithm>

void convertImagesToGray(const QStringList& imageList, QDir& directory, std::vector<KImageGray>& grayImages) {
    foreach(QString filename, imageList) {
        QImage qImage(directory.filePath(filename));

        if(qImage.isNull()) {
            continue;
        }

        KImageColor kImageColor(qImage.height(), qImage.width());
        for(int y = 0; y < qImage.height(); y++) {
            for(int x = 0; x < qImage.width(); x++) {
                QColor color = qImage.pixelColor(x, y);
                kImageColor[y][x].r = color.red();
                kImageColor[y][x].g = color.green();
                kImageColor[y][x].b = color.blue();
            }
        }

        KImageGray kImageGray(qImage.height(), qImage.width());
        for(int y = 0; y < kImageColor.Row(); y++) {
            for(int x = 0; x < kImageColor.Col(); x++) {
                kImageGray[y][x] = static_cast<unsigned char>(0.299 * kImageColor[y][x].r +
                                                              0.587 * kImageColor[y][x].g +
                                                              0.114 * kImageColor[y][x].b);
            }
        }

        grayImages.push_back(kImageGray);
    }
}

std::vector<std::pair<QString, std::vector<QRect>>> parseFDDBLabels(const QString& labelPath) {
    std::vector<std::pair<QString, std::vector<QRect>>> result;
    QFile file(labelPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        std::cout << "FDDB: cannot open " << labelPath.toStdString() << std::endl;
        return result;
    }

    QTextStream in(&file);
    QString currentImagePath;
    std::vector<QRect> currentBboxes;

    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty()) continue;

        if (line.startsWith('#')) {
            if (!currentImagePath.isEmpty()) {
                result.push_back({currentImagePath, currentBboxes});
            }
            currentImagePath = line.mid(2).trimmed();
            currentBboxes.clear();
        } else {
            QStringList parts = line.split(' ', Qt::SkipEmptyParts);
            if (parts.size() >= 4) {
                int xMin = parts[0].toInt();
                int yMin = parts[1].toInt();
                int xMax = parts[2].toInt();
                int yMax = parts[3].toInt();
                int w = xMax - xMin;
                int h = yMax - yMin;
                if (w > 0 && h > 0) {
                    currentBboxes.push_back(QRect(xMin, yMin, w, h));
                }
            }
        }
    }
    if (!currentImagePath.isEmpty()) {
        result.push_back({currentImagePath, currentBboxes});
    }

    file.close();
    return result;
}

void extractFDDBSamples(const std::vector<std::pair<QString, std::vector<QRect>>>& fddbData,
                        const QString& fddbImageRoot,
                        int maxPositives, int maxNegatives,
                        std::vector<KImageGray>& positives,
                        std::vector<KImageGray>& negatives) {
    int posCount = 0;
    int negCount = 0;
    int negsPerImage = 5;

    for (size_t imgIdx = 0; imgIdx < fddbData.size(); imgIdx++) {
        if (posCount >= maxPositives && negCount >= maxNegatives) break;

        const QString& relPath = fddbData[imgIdx].first;
        const std::vector<QRect>& bboxes = fddbData[imgIdx].second;
        QString fullPath = fddbImageRoot + "/" + relPath;
        QImage qImg(fullPath);
        if (qImg.isNull()) continue;

        int imgW = qImg.width();
        int imgH = qImg.height();

        // Extract positives
        if (posCount < maxPositives) {
            for (const QRect& bbox : bboxes) {
                if (posCount >= maxPositives) break;
                if (bbox.width() < 12 || bbox.height() < 12) continue;

                int x1 = std::max(0, bbox.x());
                int y1 = std::max(0, bbox.y());
                int x2 = std::min(imgW, bbox.x() + bbox.width());
                int y2 = std::min(imgH, bbox.y() + bbox.height());
                if (x2 - x1 < 12 || y2 - y1 < 12) continue;

                QImage cropped = qImg.copy(x1, y1, x2 - x1, y2 - y1);
                QImage scaled = cropped.scaled(24, 24, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

                KImageGray grayImg(24, 24);
                for (int py = 0; py < 24; py++) {
                    for (int px = 0; px < 24; px++) {
                        QColor c = scaled.pixelColor(px, py);
                        grayImg[py][px] = static_cast<unsigned char>(
                            0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue());
                    }
                }
                positives.push_back(grayImg);
                posCount++;
            }
        }

        // Extract negatives
        if (negCount < maxNegatives && imgW > 24 && imgH > 24) {
            for (int patchIdx = 0; patchIdx < negsPerImage; patchIdx++) {
                if (negCount >= maxNegatives) break;

                int px = (int)((imgIdx * 73 + patchIdx * 37 + 13) % (imgW - 24));
                int py = (int)((imgIdx * 53 + patchIdx * 97 + 7) % (imgH - 24));
                QRect patchRect(px, py, 24, 24);

                bool overlaps = false;
                for (const QRect& bbox : bboxes) {
                    if (computeQRectIoU_cascade(patchRect, bbox) > 0.1) {
                        overlaps = true;
                        break;
                    }
                }
                if (overlaps) continue;

                QImage patch = qImg.copy(px, py, 24, 24);
                KImageGray grayPatch(24, 24);
                for (int y = 0; y < 24; y++) {
                    for (int x = 0; x < 24; x++) {
                        QColor c = patch.pixelColor(x, y);
                        grayPatch[y][x] = static_cast<unsigned char>(
                            0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue());
                    }
                }
                negatives.push_back(grayPatch);
                negCount++;
            }
        }
    }

    std::cout << "FDDB: loaded " << positives.size() << " positives, "
              << negatives.size() << " negatives" << std::endl;
}
