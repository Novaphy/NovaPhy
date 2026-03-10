#pragma once

#include "novaphy/io/scene_types.h"

namespace novaphy {

class SceneBuilderEngine {
public:
    SceneBuildResult build_from_urdf(const UrdfModelData& urdf_model) const;
    SceneBuildResult build_from_openusd(const UsdStageData& stage) const;
};

}  // namespace novaphy
