//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "FELIX_EffectivePressure.hpp"
#include "FELIX_EffectivePressure_Def.hpp"

PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::EffectivePressure,false,true)
PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::EffectivePressure,false,false)
PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::EffectivePressure,true,true)
PHAL_INSTANTIATE_TEMPLATE_CLASS_WITH_EXTRA_ARGS(FELIX::EffectivePressure,true,false)
