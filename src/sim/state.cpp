#include "novaphy/sim/state.h"

namespace novaphy {

void SimState::init(int n, const std::vector<Transform>& initial_transforms) {
    transforms = initial_transforms;
    linear_velocities.assign(n, Vec3f::Zero());
    angular_velocities.assign(n, Vec3f::Zero());
    forces.assign(n, Vec3f::Zero());
    torques.assign(n, Vec3f::Zero());
}

void SimState::clear_forces() {
    for (auto& f : forces) f = Vec3f::Zero();
    for (auto& t : torques) t = Vec3f::Zero();
}

void SimState::apply_force(int body_index, const Vec3f& force) {
    if (body_index >= 0 && body_index < static_cast<int>(forces.size())) {
        forces[body_index] += force;
    }
}

void SimState::apply_torque(int body_index, const Vec3f& torque) {
    if (body_index >= 0 && body_index < static_cast<int>(torques.size())) {
        torques[body_index] += torque;
    }
}

}  // namespace novaphy
