#ifndef FDDB_UTILS_H
#define FDDB_UTILS_H

#include <vector>
#include <utility>
#include <QString>
#include <QRect>
#include <QImage>
#include <QDir>
#include "kfc.h"

// QImage → KImageGray 변환
void convertImagesToGray(const QStringList& imageList, QDir& directory, std::vector<KImageGray>& grayImages);

// FDDB 어노테이션 파싱
std::vector<std::pair<QString, std::vector<QRect>>> parseFDDBLabels(const QString& labelPath);

// positive/negative 샘플 추출
void extractFDDBSamples(const std::vector<std::pair<QString, std::vector<QRect>>>& fddbData,
                        const QString& fddbImageRoot,
                        int maxPositives, int maxNegatives,
                        std::vector<KImageGray>& positives,
                        std::vector<KImageGray>& negatives);

#endif // FDDB_UTILS_H
