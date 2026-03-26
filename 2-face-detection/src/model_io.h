#ifndef MODEL_IO_H
#define MODEL_IO_H

#include <QString>
#include "cascade_classifier.h"

// cascade 모델 직렬화
bool saveCascadeModel(const CascadeClassifier& cascade, const QString& path);

// cascade 모델 역직렬화
CascadeClassifier* loadCascadeModel(const QString& path);

#endif // MODEL_IO_H
