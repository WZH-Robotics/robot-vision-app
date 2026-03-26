#ifndef AXXBSOLVER_H
#define AXXBSOLVER_H

#include<Eigen/Core>
#include"type.h"
#include <QtGlobal>
//#include <glog/logging.h>

//used for hand eye calibration
class AXXBSolver
{
public:
  AXXBSolver();
  AXXBSolver(const Poses A, const Poses B):A_(A),B_(B)
  {
    Q_ASSERT_X(A_.size() == B_.size(), "AXXBSolver Constructor", "Poses A and B must have the same size");
    Q_ASSERT_X(A_.size() >= 2, "AXXBSolver Constructor", "At least two motions are required");
  }

  virtual Pose SolveX()=0;

  Poses A_;
  Poses B_;
};

#endif // AXXBSOLVER_H
