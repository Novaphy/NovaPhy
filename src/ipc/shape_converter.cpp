#include "novaphy/ipc/shape_converter.h"

#include <cmath>
#include <unordered_map>

namespace novaphy {

TetMeshData box_to_tetmesh(const Vec3f& half_extents) {
    TetMeshData mesh;

    // 8 corner vertices of the box
    const double hx = static_cast<double>(half_extents.x());
    const double hy = static_cast<double>(half_extents.y());
    const double hz = static_cast<double>(half_extents.z());

    mesh.vertices = {
        {-hx, -hy, -hz},  // 0
        { hx, -hy, -hz},  // 1
        { hx,  hy, -hz},  // 2
        {-hx,  hy, -hz},  // 3
        {-hx, -hy,  hz},  // 4
        { hx, -hy,  hz},  // 5
        { hx,  hy,  hz},  // 6
        {-hx,  hy,  hz},  // 7
    };

    // 5 tetrahedra that fill a box (standard decomposition)
    mesh.tetrahedra = {
        {0, 1, 3, 4},
        {1, 2, 3, 6},
        {1, 4, 5, 6},
        {3, 4, 6, 7},
        {1, 3, 4, 6},
    };

    return mesh;
}

namespace {

// Helper: subdivide an icosphere edge, returning the midpoint vertex index
using EdgeKey = std::pair<int, int>;
struct EdgeHash {
    size_t operator()(const EdgeKey& e) const {
        return std::hash<int>()(e.first) ^ (std::hash<int>()(e.second) << 16);
    }
};

int get_midpoint(std::vector<Eigen::Vector3d>& verts,
                 std::unordered_map<EdgeKey, int, EdgeHash>& cache,
                 int a, int b, double radius) {
    EdgeKey key = (a < b) ? EdgeKey{a, b} : EdgeKey{b, a};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    Eigen::Vector3d mid = (verts[a] + verts[b]).normalized() * radius;
    int idx = static_cast<int>(verts.size());
    verts.push_back(mid);
    cache[key] = idx;
    return idx;
}

}  // namespace

TetMeshData sphere_to_tetmesh(float radius, int subdivisions) {
    TetMeshData mesh;
    const double r = static_cast<double>(radius);

    // Start with an icosahedron
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    const double s = r / std::sqrt(1.0 + t * t);

    std::vector<Eigen::Vector3d> surface_verts = {
        {-s,  t*s, 0}, { s,  t*s, 0}, {-s, -t*s, 0}, { s, -t*s, 0},
        { 0, -s,  t*s}, { 0,  s,  t*s}, { 0, -s, -t*s}, { 0,  s, -t*s},
        { t*s, 0, -s}, { t*s, 0,  s}, {-t*s, 0, -s}, {-t*s, 0,  s},
    };

    std::vector<Eigen::Vector3i> faces = {
        {0,11,5}, {0,5,1}, {0,1,7}, {0,7,10}, {0,10,11},
        {1,5,9}, {5,11,4}, {11,10,2}, {10,7,6}, {7,1,8},
        {3,9,4}, {3,4,2}, {3,2,6}, {3,6,8}, {3,8,9},
        {4,9,5}, {2,4,11}, {6,2,10}, {8,6,7}, {9,8,1},
    };

    // Subdivision
    for (int sub = 0; sub < subdivisions; ++sub) {
        std::unordered_map<EdgeKey, int, EdgeHash> cache;
        std::vector<Eigen::Vector3i> new_faces;
        new_faces.reserve(faces.size() * 4);

        for (const auto& f : faces) {
            int a = get_midpoint(surface_verts, cache, f[0], f[1], r);
            int b = get_midpoint(surface_verts, cache, f[1], f[2], r);
            int c = get_midpoint(surface_verts, cache, f[2], f[0], r);

            new_faces.push_back({f[0], a, c});
            new_faces.push_back({f[1], b, a});
            new_faces.push_back({f[2], c, b});
            new_faces.push_back({a, b, c});
        }
        faces = std::move(new_faces);
    }

    // Build tet mesh: center vertex (0) + all surface vertices
    // Vertex 0 is the sphere center
    mesh.vertices.reserve(surface_verts.size() + 1);
    mesh.vertices.push_back(Eigen::Vector3d::Zero());  // center at index 0
    for (const auto& v : surface_verts) {
        mesh.vertices.push_back(v);
    }

    // Each surface triangle becomes a tetrahedron with the center
    // Surface vertex indices are offset by 1 (center is at 0)
    mesh.tetrahedra.reserve(faces.size());
    for (const auto& f : faces) {
        mesh.tetrahedra.push_back({0, f[0] + 1, f[1] + 1, f[2] + 1});
    }

    return mesh;
}

}  // namespace novaphy
