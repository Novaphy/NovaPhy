#pragma once

#include <string>

#include "novaphy/io/scene_types.h"

namespace novaphy {

class UrdfParser {
public:
    UrdfModelData parse_file(const std::string& urdf_path) const;
    std::string write_string(const UrdfModelData& model) const;
    void write_file(const UrdfModelData& model, const std::string& urdf_path) const;
};

}  // namespace novaphy
