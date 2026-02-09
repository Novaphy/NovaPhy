#include "novaphy/core/model_builder.h"

#include "novaphy/core/model.h"

namespace novaphy {

int ModelBuilder::add_body(const RigidBody& body, const Transform& transform) {
    int idx = static_cast<int>(bodies_.size());
    bodies_.push_back(body);
    initial_transforms_.push_back(transform);
    return idx;
}

int ModelBuilder::add_shape(const CollisionShape& shape) {
    int idx = static_cast<int>(shapes_.size());
    shapes_.push_back(shape);
    return idx;
}

int ModelBuilder::add_ground_plane(float y, float friction, float restitution) {
    CollisionShape plane = CollisionShape::make_plane(
        Vec3f(0.0f, 1.0f, 0.0f), y, friction, restitution);
    return add_shape(plane);
}

Model ModelBuilder::build() const {
    Model m;
    m.bodies = bodies_;
    m.initial_transforms = initial_transforms_;
    m.shapes = shapes_;
    return m;
}

}  // namespace novaphy
