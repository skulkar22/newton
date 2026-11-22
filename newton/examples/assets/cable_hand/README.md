# Cable-Driven Hand Visual Assets

## Overview

`cable_hand_visuals.usd` holds the visual meshes of the cable-driven hand example: a right hand with four fingers and a thumb, flanged sheaves at every joint and one linear actuator per cable in the hand. The index and the middle finger carry two cables each (a knuckle cable and a curl cable, with two actuators in line behind the finger); the ring finger, the little finger and the thumb carry one. The palm is one modeled body with a heel, a wrist stem and a flange; the thumb turns on a seat at the nose of a thenar lobe.

The meshes are for display only. The example creates its collision shapes, joints and cables from the hand specification, not from this file.

## How it is generated

The file is written by `newton/examples/cable/cable_hand_visuals.py` from the default hand specification of the example. To regenerate it, run:

```
python -m newton.examples.cable.cable_hand_visuals
```

This command reads the default hand specification from the example, so it needs Newton and Warp besides NumPy and `pxr`. A test loads the shipped file and compares its content with what the generator builds: part names, bodies, materials, vertex positions to 1 µm and triangles. Do not edit the file by hand.

## Layout

The file is a binary USD stage with `metersPerUnit = 1` and `upAxis = "Z"`. All lengths are in meters.

| Prim path | Content |
|---|---|
| `/cable_hand/<body>/<part>` | One `Mesh` per visual part, listed under the body that carries it. |
| `/cable_hand/prototypes/<name>` | Meshes that are used more than once (sheaves, actuators, carriages, screws, inner ears of the links). The parts that use them reference these prims and carry a translate and an orient operation. |
| `/cable_hand/looks/<tag>` | One preview surface per material tag with the default base color, roughness and metallic value. |

Every mesh is given in the frame of its body. The mesh has no normals. Hard edges have split vertices, so normals computed per vertex give the intended shading.

## Body frames

| Body | Frame |
|---|---|
| `palm` | Origin at the center of the wrist end of the palm. +x is distal, +y is palmar and +z points to the thumb side. |
| `thumb_column` | The thumb digit frame. The origin is on the thumb MCP axis, +x runs along the straight thumb, +y is palmar and +z is the flexion axis. The frame turns with the opposition hinge. |
| `<digit>/link1`, `<digit>/link2`, `<digit>/link3` | Proximal, middle and distal link. The origin is on the proximal joint axis of the link, +x points to the next joint, +y is palmar and +z is the flexion axis. |
| `<digit>/slider` | The actuator carriage of the digit's first (long) cable. The frame is parallel to the digit frame and coincides with it at zero travel. It moves along the pull direction of the slider. |
| `<digit>/slider_<cable>` | The carriage of every further cable of a digit, with the same frame convention: `index/slider_knuckle` and `middle/slider_knuckle`. |

`<digit>` is one of `index`, `middle`, `ring`, `little` and `thumb`.

## Material tags

`frame`, `cover`, `housing`, `tip`, `pad`, `sheave`, `guide`, `pin`, `actuator` and `carriage`. The default look of each tag is in `MATERIALS` of the generator module.

## Use

```python
from newton.examples.cable import cable_hand_visuals

parts = cable_hand_visuals.load_visuals_usd()
shapes = cable_hand_visuals.add_visuals_to_builder(builder, parts, bodies)
```

`bodies` maps each body key to a body index of the builder.

## License

Original work created for this example. Licensed under the Apache License, Version 2.0. See `LICENSE`.
