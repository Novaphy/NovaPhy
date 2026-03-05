#pragma once

#include <string>
#include <utility>
#include <vector>

#include "novaphy/core/articulation.h"
#include "novaphy/core/joint.h"
#include "novaphy/core/model.h"
#include "novaphy/math/math_types.h"

namespace novaphy {

enum class UrdfGeometryType {
    Box,
    Sphere,
    Cylinder,
    Mesh,
    Unknown
};

struct UrdfGeometry {
    UrdfGeometryType type = UrdfGeometryType::Unknown;
    Vec3f size = Vec3f::Zero();
    float radius = 0.0f;
    float length = 0.0f;
    std::string mesh_filename;
    Vec3f mesh_scale = Vec3f::Ones();
};

struct UrdfVisual {
    Transform origin = Transform::identity();
    UrdfGeometry geometry;
    std::string material_name;
};

struct UrdfCollision {
    Transform origin = Transform::identity();
    UrdfGeometry geometry;
    float friction = 0.5f;
    float restitution = 0.0f;
};

struct UrdfInertial {
    Transform origin = Transform::identity();
    float mass = 0.0f;
    Mat3f inertia = Mat3f::Zero();
};

struct UrdfLink {
    std::string name;
    UrdfInertial inertial;
    std::vector<UrdfVisual> visuals;
    std::vector<UrdfCollision> collisions;
};

struct UrdfJoint {
    std::string name;
    std::string type;
    std::string parent_link;
    std::string child_link;
    Transform origin = Transform::identity();
    Vec3f axis = Vec3f(0.0f, 0.0f, 1.0f);
    float lower_limit = 0.0f;
    float upper_limit = 0.0f;
    float effort_limit = 0.0f;
    float velocity_limit = 0.0f;
};

struct UrdfModelData {
    std::string name;
    std::vector<UrdfLink> links;
    std::vector<UrdfJoint> joints;
};

struct UsdAnimationTrack {
    std::string property_name;
    std::vector<std::pair<float, Vec3f>> vec3_samples;
    std::vector<std::pair<float, Vec4f>> vec4_samples;
};

struct UsdPrim {
    std::string path;
    std::string name;
    std::string type_name;
    std::string parent_path;
    Transform local_transform = Transform::identity();
    float mass = 0.0f;
    float density = 0.0f;
    std::string material_binding;
    Vec3f box_half_extents = Vec3f::Zero();
    float sphere_radius = 0.0f;
    std::vector<UsdAnimationTrack> tracks;
};

struct UsdStageData {
    std::string default_prim;
    std::string up_axis = "Y";
    float meters_per_unit = 1.0f;
    std::vector<UsdPrim> prims;
};

struct SceneBuildResult {
    Model model;
    Articulation articulation;
    std::vector<std::string> warnings;
};

struct RecordedKeyframe {
    float time = 0.0f;
    int body_index = -1;
    Vec3f position = Vec3f::Zero();
    Vec4f rotation = Vec4f(0.0f, 0.0f, 0.0f, 1.0f);
    Vec3f linear_velocity = Vec3f::Zero();
    Vec3f angular_velocity = Vec3f::Zero();
};

struct RecordedCollisionEvent {
    float time = 0.0f;
    int body_a = -1;
    int body_b = -1;
    Vec3f position = Vec3f::Zero();
    Vec3f normal = Vec3f::Zero();
    float penetration = 0.0f;
};

struct RecordedConstraintReaction {
    float time = 0.0f;
    std::string joint_name;
    VecXf wrench = VecXf::Zero(6);
};

}  // namespace novaphy
