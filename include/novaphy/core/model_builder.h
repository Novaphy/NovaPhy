#pragma once

#include <string>
#include <vector>

#include "novaphy/core/body.h"
#include "novaphy/core/shape.h"
#include "novaphy/math/math_types.h"

namespace novaphy {

// Forward declaration
struct Model;

/// Builder for constructing a physics Model.
/// Accumulates bodies and shapes, then produces an immutable Model.
class ModelBuilder {
public:
    ModelBuilder() = default;

    /// Add a dynamic rigid body. Returns body index.
    int add_body(const RigidBody& body, const Transform& transform = Transform::identity());

    /// Add a collision shape attached to a body. Returns shape index.
    int add_shape(const CollisionShape& shape);

    /// Add an infinite ground plane. Returns shape index.
    int add_ground_plane(float y = 0.0f, float friction = 0.5f, float restitution = 0.0f);

    /// Build the immutable Model from accumulated data.
    Model build() const;

    // Accessors for inspection
    int num_bodies() const { return static_cast<int>(bodies_.size()); }
    int num_shapes() const { return static_cast<int>(shapes_.size()); }

private:
    std::vector<RigidBody> bodies_;
    std::vector<Transform> initial_transforms_;
    std::vector<CollisionShape> shapes_;
};

}  // namespace novaphy
