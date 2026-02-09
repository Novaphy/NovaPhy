#pragma once

#include <vector>

#include "novaphy/core/body.h"
#include "novaphy/core/shape.h"
#include "novaphy/math/math_types.h"

namespace novaphy {

/// Immutable physics model: describes the scene (bodies, shapes, initial state).
/// Created via ModelBuilder::build().
struct Model {
    std::vector<RigidBody> bodies;
    std::vector<Transform> initial_transforms;
    std::vector<CollisionShape> shapes;

    int num_bodies() const { return static_cast<int>(bodies.size()); }
    int num_shapes() const { return static_cast<int>(shapes.size()); }
};

}  // namespace novaphy
