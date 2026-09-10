"""Terrain for the box backflip: a rigid floor, a box to stand on, and a soft
crash mat below the edge the duck flips off.

Local frame of the tile: the generator places each tile by its CORNER, so
local x runs 0..size[0] and local y runs 0..size[1]; everything here sits at
y = size[1]/2 so that, with one tile, it is centred on the world origin (the
out-of-terrain-bounds truncation is measured from the world origin). The box top is at z = ``box_height``; its -x edge —
the one the duck flips off — is at x = ``edge_x``. The mat lies in front of
that edge (x < edge_x) on the floor. The spawn origin is ON the box top, a few
centimetres inside the edge, so the duck starts standing on the box and has
to produce the whole move itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

BOX_HEIGHT = 0.8      # m — 0.7–0.9 completes a full turn in the open-loop study (2026-09-09)
BOX_LENGTH = 1.2      # m along x
MAT_THICKNESS = 0.12  # m
MAT_LENGTH = 1.6      # m along x, in front of the edge
EDGE_X = 2.0          # box edge position in the tile's local frame
SPAWN_INSET = 0.03    # spawn this far inside the edge (feet centre)


@dataclass(kw_only=True)
class BoxFlipTerrainCfg(SubTerrainCfg):
    box_height: float = BOX_HEIGHT
    box_length: float = BOX_LENGTH
    mat_thickness: float = MAT_THICKNESS
    mat_length: float = MAT_LENGTH
    edge_x: float = EDGE_X
    spawn_inset: float = SPAWN_INSET
    floor_thickness: float = 0.5

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
        body = spec.body("terrain")
        L, W = self.size
        t = self.floor_thickness
        yc = W / 2.0
        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(L / 2.0, W / 2.0, t / 2.0),
            pos=(L / 2.0, yc, -t / 2.0),
        )
        box = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.box_length / 2.0, W / 2.0, self.box_height / 2.0),
            pos=(self.edge_x + self.box_length / 2.0, yc, self.box_height / 2.0),
        )
        box.friction = np.array([1.0, 0.005, 0.0001])
        mat = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.mat_length / 2.0, W / 2.0, self.mat_thickness / 2.0),
            pos=(self.edge_x - 0.02 - self.mat_length / 2.0, yc, self.mat_thickness / 2.0),
        )
        # A firm crash mat: soft-ish contact (2 cm-ish give), grippy.
        mat.solref = np.array([0.02, 1.0])
        mat.friction = np.array([1.2, 0.005, 0.0001])
        origin = np.array([self.edge_x + self.spawn_inset, yc, self.box_height])
        return TerrainOutput(
            origin=origin,
            geometries=[
                TerrainGeometry(geom=floor, color=(0.5, 0.5, 0.5, 1.0)),
                TerrainGeometry(geom=box, color=(0.55, 0.35, 0.18, 1.0)),
                TerrainGeometry(geom=mat, color=(0.2, 0.4, 0.85, 1.0)),
            ],
        )
