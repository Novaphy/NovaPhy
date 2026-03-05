import json
import time
from pathlib import Path

import numpy as np

import novaphy


def run_benchmark(steps=300, dt=1.0 / 120.0):
    builder = novaphy.ModelBuilder()
    builder.add_ground_plane(y=0.0)
    half = np.array([0.2, 0.2, 0.2], dtype=np.float32)

    side = 10
    layers = 10
    count = side * side * layers
    for k in range(layers):
        for i in range(side):
            for j in range(side):
                body = novaphy.RigidBody.from_box(1.0, half)
                pos = np.array(
                    [i * 0.45 - 2.0, 0.5 + k * 0.45, j * 0.45 - 2.0], dtype=np.float32
                )
                idx = builder.add_body(body, novaphy.Transform.from_translation(pos))
                builder.add_shape(novaphy.CollisionShape.make_box(half, idx))

    world = novaphy.World(builder.build())
    start = time.perf_counter()
    for _ in range(steps):
        world.step(dt)
    elapsed = time.perf_counter() - start
    fps = steps / elapsed
    return {"bodies": count, "steps": steps, "dt": dt, "elapsed_sec": elapsed, "fps": fps}


if __name__ == "__main__":
    result = run_benchmark()
    output = Path("build") / "benchmark_rigid_1000.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
