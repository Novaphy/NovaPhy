#include "novaphy/dynamics/integrator.h"

namespace novaphy {

void SymplecticEuler::integrate_velocity(Vec3f& linear_vel, Vec3f& angular_vel,
                                          const Vec3f& force, const Vec3f& torque,
                                          float inv_mass, const Mat3f& inv_inertia,
                                          const Vec3f& gravity, float dt) {
    // Semi-implicit Euler: v(t+dt) = v(t) + a * dt
    linear_vel += (gravity + force * inv_mass) * dt;
    angular_vel += inv_inertia * torque * dt;

    // Angular damping to prevent energy blow-up
    angular_vel *= 0.98f;
}

void SymplecticEuler::integrate_position(Transform& transform,
                                          const Vec3f& linear_vel,
                                          const Vec3f& angular_vel, float dt) {
    // Position update
    transform.position += linear_vel * dt;

    // Orientation update via quaternion derivative
    // dq/dt = 0.5 * omega * q
    Quatf& q = transform.rotation;
    Quatf omega_quat(0, angular_vel.x(), angular_vel.y(), angular_vel.z());
    Quatf dq;
    dq.w() = 0.5f * (-angular_vel.x() * q.x() - angular_vel.y() * q.y() - angular_vel.z() * q.z());
    dq.x() = 0.5f * (angular_vel.x() * q.w() + angular_vel.z() * q.y() - angular_vel.y() * q.z());
    dq.y() = 0.5f * (angular_vel.y() * q.w() - angular_vel.z() * q.x() + angular_vel.x() * q.z());
    dq.z() = 0.5f * (angular_vel.z() * q.w() + angular_vel.y() * q.x() - angular_vel.x() * q.y());

    q.w() += dq.w() * dt;
    q.x() += dq.x() * dt;
    q.y() += dq.y() * dt;
    q.z() += dq.z() * dt;
    q.normalize();
}

}  // namespace novaphy
