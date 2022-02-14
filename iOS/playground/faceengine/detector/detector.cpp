#include "detector.h"
#include "./centerface/centerface.h"
#include "./ultraface/ultraface.h"

namespace mirror {
Detector *CenterfaceFactory::CreateDetecter() { return new Centerface(); }

Detector *UltrafaceFactory::CreateDetecter() { return new UltraFace(); }

} // namespace mirror
