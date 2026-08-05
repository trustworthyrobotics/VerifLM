from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np

from pipeline.core import ValidationProgram, mat_to_quat_wxyz


# Object descriptions, assets, physics properties, and configured grasps.
OBJECT_CATALOG = {
    'wooden_rack': {
        'prompt': 'wooden rack',
        'mesh': 'models/meshes/wooden_rack/wooden_rack.obj',
        'dimensions': (0.2000, 0.1003, 0.0551),
        'asset_xml': 'meshes/wooden_rack_remeshed/wooden_rack_asset.xml',
        'body_xml': 'meshes/wooden_rack_remeshed/wooden_rack_body.xml',
        'geom_friction': '0.05 0.001 0.0001',
        'geom_condim': '4',
    },
    'wooden_stick_toy': {
        'prompt': 'wooden stick toy',
        'mesh': 'models/meshes/wooden_stick_toy/wooden_stick_toy.obj',
        'dimensions': (0.1429, 0.1005, 0.2067),
        'asset_xml': 'meshes/wooden_stick_toy_remeshed/wooden_stick_toy_asset.xml',
        'body_xml': 'meshes/wooden_stick_toy_remeshed/wooden_stick_toy_body.xml',
        'geom_friction': '0.8 0.02 0.001',
    },
    'red_mug_small': {
        'prompt': 'red mug small',
        'mesh': 'models/meshes/red_mug_small/red_mug_small.obj',
        'dimensions': (0.0564, 0.0729, 0.0567),
        'asset_xml': 'meshes/red_mug_small_remeshed/red_mug_small_asset.xml',
        'body_xml': 'meshes/red_mug_small_remeshed/red_mug_small_body.xml',
        'grasps': [
            {
                'offset_xyz': (0.0, 0.0, 0.25),
                'offset_rpy_deg': (180.0, 0.0, 0.0),
                'hover_offset_z': 0.1,
                'close_ctrl_value': 0.025,
            },
        ],
    },
    'red_mug_big': {
        'prompt': 'red mug',
        'mesh': 'models/meshes/red_mug_big/red_mug_big.obj',
        'dimensions': (0.0879, 0.1108, 0.0876),
        'asset_xml': 'meshes/red_mug_big_remeshed/red_mug_big_asset.xml',
        'body_xml': 'meshes/red_mug_big_remeshed/red_mug_big_body.xml',
        'geom_friction': '0.8 0.02 0.001',
        'grasps': [
            {
                'offset_xyz': (-0.041, 0, 0.17),
                'offset_rpy_deg': (180.0, 0.0, 0),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.035,
            },
            {
                'offset_xyz': (0.041, 0, 0.17),
                'offset_rpy_deg': (180.0, 0.0, 0),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.035,
            },
            {
                'offset_xyz': (0.0, -0.041, 0.17),
                'offset_rpy_deg': (180.0, 0.0, -90),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.035,
            },
            {
                'offset_xyz': (0.0, 0.041, 0.17),
                'offset_rpy_deg': (180.0, 0.0, -90),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.035,
            },
            {
                'offset_xyz': (0.0, -0.041, 0.17),
                'offset_rpy_deg': (180.0, 0.0, 90),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.035,
            },
            {
                'offset_xyz': (0.0, 0.041, 0.17),
                'offset_rpy_deg': (180.0, 0.0, 90),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.035,
            },
        ],
    },
    'blue_mug_big': {
        'prompt': 'blue mug',
        'mesh': 'models/meshes/blue_mug_big/blue_mug_big.obj',
        'dimensions': (0.0849, 0.1174, 0.0918),
        'asset_xml': 'meshes/blue_mug_big_remeshed/blue_mug_big_asset.xml',
        'body_xml': 'meshes/blue_mug_big_remeshed/blue_mug_big_body.xml',
        'grasps': [
            {
                'offset_xyz': (0.0, -0.037, 0.13),
                'offset_rpy_deg': (180.0, 0.0, 90),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.035,
            },
            {
                'offset_xyz': (0.0, 0.037, 0.11),
                'offset_rpy_deg': (180.0, 0.0, 90),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.035,
            },
            {
                'offset_xyz': (-0.037, 0, 0.11),
                'offset_rpy_deg': (180.0, 0.0, 0),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.035,
            },
            {
                'offset_xyz': (0.037, 0, 0.11),
                'offset_rpy_deg': (180.0, 0.0, 0),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.045,
            },
        ],
    },
    'green_plate_small': {
        'prompt': 'green plate',
        'mesh': 'models/meshes/green_plate_small/green_plate_small.obj',
        'dimensions': (0.1071, 0.1059, 0.0176),
        'primitive': 'small_dish',
        'grasps': [
            {
                'offset_xyz': (0.0, 0.16, 0.01),
                'offset_rpy_deg': (90.0, -90.0, 0),
                'hover_offset_xyz': (0.0, 0.1, 0.0),
                'close_ctrl_value': 0.045,
            },
        ],
    },
    'pink_plate_small': {
        'prompt': 'pink plate',
        'mesh': 'models/meshes/pink_plate_small/pink_plate_small.obj',
        'dimensions': (0.1076, 0.1078, 0.0180),
        'primitive': 'small_dish',
        'grasps': [
            {
                'offset_xyz': (0.0, 0.16, 0.01),
                'offset_rpy_deg': (90.0, -90.0, 0),
                'hover_offset_xyz': (0.0, 0.05, 0.0),
                'close_ctrl_value': 0.045,
            },
        ],
    },
    'yellow_plate_small': {
        'prompt': 'yellow plate',
        'mesh': 'models/meshes/yellow_plate_small/yellow_plate_small.obj',
        'dimensions': (0.1069, 0.1071, 0.0145),
        'primitive': 'small_dish',
        'grasps': [
            {
                'offset_xyz': (0.0, 0.16, 0.01),
                'offset_rpy_deg': (90.0, 90.0, 0),
                'hover_offset_xyz': (0.0, 0.05, 0.0),
                'close_ctrl_value': 0.045,
            },
        ],
    },
    'red_plate_big': {
        'prompt': 'red plate',
        'mesh': 'models/meshes/red_plate_big/red_plate_big.obj',
        'dimensions': (0.1208, 0.1248, 0.0145),
        'primitive': 'big_dish',
        'grasps': [
            {
                'offset_xyz': (0.0, 0.14, 0.0),
                'offset_rpy_deg': (90.0, 90.0, 0),
                'hover_offset_xyz': (0.0, 0.1, 0.0),
                'close_ctrl_value': 0.035,
            },
        ],
    },
    'blue_plate_big': {
        'prompt': 'blue plate',
        'mesh': 'models/meshes/blue_plate_big/blue_plate_big.obj',
        'dimensions': (0.1245, 0.1226, 0.0156),
        'primitive': 'big_dish',
        'grasps': [
            {
                'offset_xyz': (0.0, 0.14, 0.0),
                'offset_rpy_deg': (90.0, 90.0, 0),
                'hover_offset_xyz': (0.0, 0.1, 0.0),
                'close_ctrl_value': 0.035,
            },
        ],
    },
    'teapot': {
        'prompt': 'teapot',
        'mesh': 'models/meshes/teapot/teapot.obj',
        'dimensions': (0.1676, 0.0919, 0.0961),
        'asset_xml': 'meshes/teapot_remeshed/teapot_asset.xml',
        'body_xml': 'meshes/teapot_remeshed/teapot_body.xml',
        'geom_friction': '0.15 0.001 0.0005',
        'grasps': [
            {
                'offset_xyz': (0.075, 0.0, 0.17),
                'offset_rpy_deg': (180.0, 0.0, -90.0),
                'hover_offset_z': 0.1,
                'close_ctrl_value': 0.035,
            },
            {
                'offset_xyz': (0.075, 0.0, 0.17),
                'offset_rpy_deg': (180.0, 0.0, 90.0),
                'hover_offset_z': 0.1,
                'close_ctrl_value': 0.035,
            },
        ],
    },
    'yellow_cap': {
        'prompt': 'yellow cap',
        'mesh': 'models/meshes/yellow_cap/yellow_cap.obj',
        'dimensions': (0.0743, 0.0757, 0.0278),
        'asset_xml': 'meshes/yellow_cap_remeshed/yellow_cap_asset.xml',
        'body_xml': 'meshes/yellow_cap_remeshed/yellow_cap_body.xml',
        'geom_friction': '0.15 0.001 0.0005',
        'geom_solimp': '0.9 0.95 0.001 0.5 2',
        'grasps': [
            {
                'offset_xyz': (0.0, 0.0, 0.133),
                'offset_rpy_deg': (180.0, 0.0, 0.0),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.025,
            },
        ],
    },
    'red_cup': {
        'prompt': 'red cup',
        'mesh': 'models/meshes/red_cup/red_cup.obj',
        'dimensions': (0.0714, 0.0728, 0.1085),
        'primitive': 'cup',
        'grasps': [
            {
                'offset_xyz': (0.0, 0.03, 0.22),
                'offset_rpy_deg': (180.0, 0.0, 90.0),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.025,
            },
            {
                'offset_xyz': (0.0, 0.03, 0.22),
                'offset_rpy_deg': (180.0, 0.0, -90.0),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.025,
            },
        ],
    },
    'green_cup': {
        'prompt': 'green cup',
        'mesh': 'models/meshes/green_cup/green_cup.obj',
        'dimensions': (0.0729, 0.0736, 0.1115),
        'primitive': 'small_cup',
        'grasps': [
            {
                'offset_xyz': (0.0, 0.03, 0.22),
                'offset_rpy_deg': (180.0, 0.0, 90.0),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.025,
            },
            {
                'offset_xyz': (0.0, 0.03, 0.22),
                'offset_rpy_deg': (180.0, 0.0, -90.0),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.025,
            },
        ],
    },
    'blue_cup': {
        'prompt': 'blue cup',
        'mesh': 'models/meshes/blue_cup/blue_cup.obj',
        'dimensions': (0.0728, 0.0723, 0.1104),
        'primitive': 'large_cup',
        'grasps': [
            {
                'offset_xyz': (0.0, 0.03, 0.22),
                'offset_rpy_deg': (180.0, 0.0, 90.0),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.025,
            },
            {
                'offset_xyz': (0.0, 0.03, 0.22),
                'offset_rpy_deg': (180.0, 0.0, -90.0),
                'hover_offset_z': 0.08,
                'close_ctrl_value': 0.025,
            },
        ],
    },
}
# Default benchmark and camera used by the CLI.
BENCHMARK = "plates_on_rack"
CAMERA_TYPE = "gemini"

# VLM provider for each pipeline operation that actually uses a model.
# Goal-image generation remains Gemini-only in vlm.py.
VLM_PROVIDERS = {
    "step_1": "openai",   # optional task-stage decomposition
    "step_6": "openai",   # sampling-axis reasoning
    "validation_program": "openai",  # candidate-validation semantics
    "step_9": "gemini",   # candidate selection
}

# Shared MuJoCo tabletop Z coordinate. Object source poses below are expressed
# relative to this value so changing the table does not leave static objects
# floating above or embedded in it.
SURFACE_HEIGHT_M = 0.779


# Benchmark definitions contain only values that differ by task.
# Position drift is checked in world coordinates. Express the same absolute cap
# on every axis as a fraction of the active object's corresponding dimension.
# Orientation fallback values below retain the legacy max_ori_change unit
# interpretation. They are rounded for readability, without unit conversion.
def _world_drift_fractions(cap_m: float, active_object_id: str) -> list[float]:
    return [
        cap_m / dimension
        for dimension in OBJECT_CATALOG[active_object_id]["dimensions"]
    ]


BENCHMARKS = {
    'mug_on_plate': {
        'task': {
            'objects': {
                'red_mug_big': {'source_pose': {'xyz': (-0.2, 0.4, SURFACE_HEIGHT_M + 0.0407), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'blue_plate_big': {'source_pose': {'xyz': (0, 0, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
            },
            'default_stages': [
                {
                    'active': 'red_mug_big',
                    'passive': 'blue_plate_big',
                    'instruction': 'Place the red mug on the blue plate.',
                },
            ],
        },
        'motion': {
            'allowed_penetration': 0.006,
            'z_rotation_samples': 1,
            'post_release_hover_m': 0.08,
            'preinsertion': {
                'use_preinsertion': True,
                'axis': '+z',
                'offset_m': 0.1,
                'position_range': np.array([0.04, 0.04, 0.04]),
                'position_samples': 25,
                'waypoint_allowance': 3,
                'allowed_penetration': 0.001,
            },
        },
        'digital_twin': {
            'sampling': {
                'n_samples': 1000,
                'normal_position_range': np.array([0.03, 0.03, 0.03]),
                'normal_orientation_range': np.array([60.0, 60.0, 60.0]),
                'sim_steps': 500,
            },
            'validation_program': {
                'validation_mode': 'settling',
                'initial_max_pen': 0.01,
                'final_max_pen': 0.001,
                'required_contacts': ['active_passive'],
                'forbidden_contacts': ['active_table'],
                'xyz_relations': [{
                    'axis': 'z',
                    'relation': 'greater_than',
                    'subject': 'active',
                    'reference': 'passive',
                }],
                'allowed_position_drift_fraction': _world_drift_fractions(
                    0.05, 'red_mug_big'
                ),
                # Preserve the legacy unit interpretation at readable precision.
                'allowed_orientation_drift_deg': 0.07,
                'reasoning': 'The mug must remain supported above the plate without touching the table.',
            },
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (-1, -1, 2), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (1, 1, 2), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
        ),
    },
    'mugs_on_plates': {
        'task': {
            'objects': {
                'red_mug_big': {'source_pose': {'xyz': (-0.2, 0.4, SURFACE_HEIGHT_M + 0.0407), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'red_plate_big': {'source_pose': {'xyz': (0.0, 0.0, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'blue_mug_big': {'source_pose': {'xyz': (0.2, 0.4, SURFACE_HEIGHT_M + 0.0407), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'blue_plate_big': {'source_pose': {'xyz': (0.4, 0.0, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
            },
            'default_stages': [
                {
                    'active': 'red_mug_big',
                    'passive': 'red_plate_big',
                    'instruction': 'Place the red mug on the red plate.',
                },
                {
                    'active': 'blue_mug_big',
                    'passive': 'blue_plate_big',
                    'instruction': 'Place the blue mug on the blue plate.',
                    'validation_program': {
                        'validation_mode': 'settling',
                        'initial_max_pen': 0.01,
                        'final_max_pen': 0.001,
                        'required_contacts': ['active_passive'],
                        'forbidden_contacts': ['active_table'],
                        'xyz_relations': [{
                            'axis': 'z',
                            'relation': 'greater_than',
                            'subject': 'active',
                            'reference': 'passive',
                        }],
                        'allowed_position_drift_fraction': _world_drift_fractions(
                            0.05, 'blue_mug_big'
                        ),
                        'allowed_orientation_drift_deg': 0.07,
                        'reasoning': 'The mug must remain supported above the plate without touching the table.',
                    },
                },
            ],
        },
        'motion': {
            'allowed_penetration': 0.006,
            'z_rotation_samples': 1,
            'post_release_hover_m': 0.08,
            'preinsertion': {
                'use_preinsertion': True,
                'axis': '+z',
                'offset_m': 0.1,
                'position_range': np.array([0.04, 0.04, 0.04]),
                'position_samples': 25,
                'waypoint_allowance': 3,
                'allowed_penetration': 0.001,
            },
        },
        'digital_twin': {
            'sampling': {
                'n_samples': 1000,
                'normal_position_range': np.array([0.03, 0.03, 0.03]),
                'normal_orientation_range': np.array([10.0, 10.0, 10.0]),
                'sim_steps': 500,
            },
            'validation_program': {
                'validation_mode': 'settling',
                'initial_max_pen': 0.01,
                'final_max_pen': 0.001,
                'required_contacts': ['active_passive'],
                'forbidden_contacts': ['active_table'],
                'xyz_relations': [{
                    'axis': 'z',
                    'relation': 'greater_than',
                    'subject': 'active',
                    'reference': 'passive',
                }],
                'allowed_position_drift_fraction': _world_drift_fractions(
                    0.05, 'red_mug_big'
                ),
                # Preserve the legacy unit interpretation at readable precision.
                'allowed_orientation_drift_deg': 0.07,
                'reasoning': 'Each mug must remain supported above its plate without touching the table.',
            },
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (-1, -1, 2), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (1, 1, 2), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
        ),
    },
    'stack_cups': {
        'task': {
            'objects': {
                'red_cup': {'source_pose': {'xyz': (0.18, 0.22, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'green_cup': {'source_pose': {'xyz': (0.0, 0.22, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'blue_cup': {'source_pose': {'xyz': (0.18, 0.22, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
            },
            'default_stages': [
                {
                    'active': 'red_cup',
                    'passive': 'blue_cup',
                    'instruction': 'Place the red cup inside the blue cup.',
                },
                {
                    'active': 'green_cup',
                    'passive': 'red_cup',
                    'instruction': 'Place the green cup inside the red cup.',
                    'validation_program': {
                        'validation_mode': 'settling',
                        'initial_max_pen': 0.08,
                        'final_max_pen': 0.001,
                        'required_contacts': ['active_passive'],
                        'forbidden_contacts': ['active_table'],
                        'xyz_relations': [],
                        'allowed_position_drift_fraction': _world_drift_fractions(
                            0.1, 'green_cup'
                        ),
                        'allowed_orientation_drift_deg': 30.0,
                        'reasoning': 'The nested cup must settle against the receiving cup without touching the table.',
                    },
                },
            ],
        },
        'motion': {
            'allowed_penetration': 0.008,
            'z_rotation_samples': 8,
            'post_release_hover_m': 0.06,
            'preinsertion': {
                'use_preinsertion': True,
                'axis': '+z',
                'offset_m': 0.15,
                'position_range': np.array([0.03, 0.03, 0.02]),
                'position_samples': 40,
                'waypoint_allowance': 3,
                'allowed_penetration': 0.008,
            },
        },
        'digital_twin': {
            'sampling': {
                'n_samples': 2000,
                'normal_position_range': np.array([0.03, 0.03, 0.03]),
                'normal_orientation_range': np.array([60.0, 60.0, 60.0]),
                'sim_steps': 500,
            },
            'validation_program': {
                'validation_mode': 'settling',
                'initial_max_pen': 0.08,
                'final_max_pen': 0.001,
                'required_contacts': ['active_passive'],
                'forbidden_contacts': ['active_table'],
                # Body-origin Z ordering is not reliable for a cup nested inside another cup.
                'xyz_relations': [],
                'allowed_position_drift_fraction': _world_drift_fractions(
                    0.1, 'red_cup'
                ),
                'allowed_orientation_drift_deg': 30.0,
                'reasoning': 'The nested cup must settle against the receiving cup without touching the table.',
            },
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -1.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
    },
    'plate_on_rack': {
        'task': {
            'objects': {
                'green_plate_small': {'source_pose': {'xyz': (-0.0150, 0.2750, SURFACE_HEIGHT_M + 0.08), 'rpy_deg': (-90.88, -9.96, 95.0)}},
                'wooden_rack': {'source_pose': {'xyz': (-0.04, 0.49, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 90.0)}},
            },
            'default_stages': [
                {
                    'active': 'green_plate_small',
                    'passive': 'wooden_rack',
                    'instruction': 'Insert the green plate into a pair of opposing slots.',
                },
            ],
        },
        'motion': {
            'allowed_penetration': 0.004,
            'z_rotation_samples': 24,
            'skip_settlement': True,
            'preinsertion': {
                'use_preinsertion': True,
                'axis': '+z',
                'offset_m': 0.08,
                'position_range': np.array([0.02, 0.02, 0.0]),
                'position_samples': 30,
                'waypoint_allowance': 5,
                'allowed_penetration': 0.004,
            },
        },
        'digital_twin': {
            'sampling': {
                'n_samples': 1000,
                'normal_position_range': np.array([0.03, 0.005, 0.02]),
                'normal_orientation_range': np.array([60.0, 60.0, 60.0]),
                'sim_steps': 600,
                'pos_bias': {'x': 'negative', 'y': None, 'z': 'negative'},
            },
            'validation_program': {
                'validation_mode': 'settling',
                'initial_max_pen': 0.2,
                'final_max_pen': 0.001,
                'required_contacts': ['active_passive'],
                'forbidden_contacts': ['active_table'],
                'xyz_relations': [{
                    'axis': 'z',
                    'relation': 'greater_than',
                    'subject': 'active',
                    'reference': 'passive',
                }],
                'allowed_position_drift_fraction': _world_drift_fractions(
                    0.08, 'green_plate_small'
                ),
                'allowed_orientation_drift_deg': 22.0,
                'reasoning': 'The plate must remain supported in the elevated rack slots without touching the table.',
            },
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.4, 0.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 0.9},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -0.4, 0.4), 'up': (0.0, 0.0, 1.0), 'zoom': 0.8},
        ),
    },
    'plates_on_rack': {
        'task': {
            'objects': {
                'green_plate_small': {'source_pose': {'xyz': (-0.2, 0.75, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'yellow_plate_small': {'source_pose': {'xyz': (0.05, 0.75, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'wooden_rack': {'source_pose': {'xyz': (-0.12, 0.6, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
            },
            'default_stages': [
                {
                    'active': 'green_plate_small',
                    'passive': 'wooden_rack',
                    'instruction': 'Insert the green plate vertically into a pair of opposing slots in the back of the wooden rack.',
                },
                {
                    'active': 'yellow_plate_small',
                    'passive': 'wooden_rack',
                    'instruction': 'Insert the yellow plate vertically into an adjacent pair of slots in the back of the wooden rack. The yellow plate should be placed in front of the green plate.',
                    'validation_program': {
                        'validation_mode': 'settling',
                        'initial_max_pen': 0.1,
                        'final_max_pen': 0.001,
                        'required_contacts': ['active_passive'],
                        'forbidden_contacts': ['active_table'],
                        'xyz_relations': [{
                            'axis': 'z',
                            'relation': 'greater_than',
                            'subject': 'active',
                            'reference': 'passive',
                        }],
                        'allowed_position_drift_fraction': _world_drift_fractions(
                            0.08, 'yellow_plate_small'
                        ),
                        'allowed_orientation_drift_deg': 8.0,
                        'reasoning': 'The plate must remain supported in the elevated rack slots without touching the table.',
                    },
                },
            ],
        },
        'motion': {
            'allowed_penetration': 0.004,
            'z_rotation_samples': 24,
            'skip_settlement': True,
            'preinsertion': {
                'use_preinsertion': True,
                'axis': '+z',
                'offset_m': 0.15,
                'position_range': np.array([0.02, 0.02, 0.0]),
                'position_samples': 30,
                'waypoint_allowance': 3,
                'allowed_penetration': 0.004,
            },
        },
        'digital_twin': {
            'sampling': {
                'n_samples': 1000,
                'normal_position_range': np.array([0.02, 0.02, 0.02]),
                'normal_orientation_range': np.array([60.0, 60.0, 60.0]),
                'sim_steps': 500,
            },
            'validation_program': {
                'validation_mode': 'settling',
                'initial_max_pen': 0.1,
                'final_max_pen': 0.001,
                'required_contacts': ['active_passive'],
                'forbidden_contacts': ['active_table'],
                'xyz_relations': [{
                    'axis': 'z',
                    'relation': 'greater_than',
                    'subject': 'active',
                    'reference': 'passive',
                }],
                'allowed_position_drift_fraction': _world_drift_fractions(
                    0.08, 'green_plate_small'
                ),
                'allowed_orientation_drift_deg': 8.0,
                'reasoning': 'Each plate must remain supported in the elevated rack slots without touching the table.',
            },
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 1.5), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -0.8, 0.8), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
    },
    'stack_plate': {
        # This setup calibrates the passive plate from the perceived board
        # frame instead of forcing its object origin onto the tabletop.
        'scene': {'anchor_passive_to_surface': False},
        'task': {
            'objects': {
                'yellow_plate_small': {'source_pose': {'xyz': (-0.1, 0.65, SURFACE_HEIGHT_M + 0.07), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'green_plate_small': {'source_pose': {'xyz': (0.1567, 0.3621, SURFACE_HEIGHT_M + 0.2309), 'rpy_deg': (0.0, 90.0, 0.0)}},
            },
            'default_stages': [
                {
                    'active': 'green_plate_small',
                    'passive': 'yellow_plate_small',
                    'instruction': 'Stack the green plate on top of the yellow plate. Make sure the green plate is on top of the yellow plate with some vertical offset so the yellow plate is still visible.',
                },
            ],
        },
        'motion': {
            'allowed_penetration': 0.004,
            'z_rotation_samples': 8,
            'post_release_hover_m': 0.06,
            'skip_settlement': True,
            'preinsertion': {
                'use_preinsertion': True,
                'axis': '+z',
                'offset_m': 0.05,
                'position_range': np.array([0.02, 0.02, 0.0]),
                'position_samples': 30,
                'waypoint_allowance': 15,
                'allowed_penetration': 0.004,
                'stop_short_waypoints': 15,
            },
        },
        'digital_twin': {
            'sampling': {
                'n_samples': 1000,
                'normal_position_range': np.array([0.02, 0.02, 0.04]),
                'normal_orientation_range': np.array([60.0, 60.0, 60.0]),
                'sim_steps': 500,
            },
            'validation_program': {
                'validation_mode': 'settling',
                'initial_max_pen': 0.08,
                'final_max_pen': 0.001,
                'required_contacts': ['active_passive'],
                'forbidden_contacts': ['active_table'],
                'xyz_relations': [{
                    'axis': 'z',
                    'relation': 'greater_than',
                    'subject': 'active',
                    'reference': 'passive',
                }],
                'allowed_position_drift_fraction': _world_drift_fractions(
                    0.05, 'green_plate_small'
                ),
                'allowed_orientation_drift_deg': 8.0,
                'reasoning': 'The upper plate must remain supported above the lower plate without touching the table.',
            },
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -1.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
    },
    'stack_plates': {
        'task': {
            'objects': {
                'green_plate_small': {'source_pose': {'xyz': (-0.2, 0.65, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'yellow_plate_small': {'source_pose': {'xyz': (0.0, 0.65, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'pink_plate_small': {'source_pose': {'xyz': (0.2, 0.65, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
            },
            'default_stages': [
                {
                    'active': 'yellow_plate_small',
                    'passive': 'green_plate_small',
                    'instruction': 'Stack the yellow plate on top of the green plate.',
                },
                {
                    'active': 'pink_plate_small',
                    'passive': 'yellow_plate_small',
                    'instruction': 'Stack the pink plate on top of the yellow plate.',
                    'validation_program': {
                        'validation_mode': 'settling',
                        'initial_max_pen': 0.08,
                        'final_max_pen': 0.001,
                        'required_contacts': ['active_passive'],
                        'forbidden_contacts': ['active_table'],
                        'xyz_relations': [{
                            'axis': 'z',
                            'relation': 'greater_than',
                            'subject': 'active',
                            'reference': 'passive',
                        }],
                        'allowed_position_drift_fraction': _world_drift_fractions(
                            0.05, 'pink_plate_small'
                        ),
                        'allowed_orientation_drift_deg': 8.0,
                        'reasoning': 'The upper plate must remain supported above the lower plate without touching the table.',
                    },
                },
            ],
        },
        'motion': {
            'allowed_penetration': 0.004,
            'z_rotation_samples': 8,
            'post_release_hover_m': 0.06,
            'preinsertion': {
                'use_preinsertion': True,
                'axis': '+z',
                'offset_m': 0.05,
                'position_range': np.array([0.02, 0.02, 0.0]),
                'position_samples': 30,
                'waypoint_allowance': 15,
                'allowed_penetration': 0.004,
            },
        },
        'digital_twin': {
            'sampling': {
                'n_samples': 1000,
                'normal_position_range': np.array([0.03, 0.03, 0.03]),
                'normal_orientation_range': np.array([60.0, 60.0, 60.0]),
                'sim_steps': 500,
            },
            'validation_program': {
                'validation_mode': 'settling',
                'initial_max_pen': 0.08,
                'final_max_pen': 0.001,
                'required_contacts': ['active_passive'],
                'forbidden_contacts': ['active_table'],
                'xyz_relations': [{
                    'axis': 'z',
                    'relation': 'greater_than',
                    'subject': 'active',
                    'reference': 'passive',
                }],
                'allowed_position_drift_fraction': _world_drift_fractions(
                    0.05, 'yellow_plate_small'
                ),
                'allowed_orientation_drift_deg': 8.0,
                'reasoning': 'Each upper plate must remain supported above the lower plate without touching the table.',
            },
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -1.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
    },
    'cap_on_teapot': {
        'scene': {
            'extra_allowed_contact_pair': ('arm1_:teapot',),
        },
        'task': {
            'objects': {
                'yellow_cap': {'source_pose': {'xyz': (-0.1100, 0.2000, SURFACE_HEIGHT_M + 0.10), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'teapot': {'source_pose': {'xyz': (-0.12, 0.2, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
            },
            'default_stages': [
                {
                    'active': 'yellow_cap',
                    'passive': 'teapot',
                    'instruction': 'Place the yellow cap on the teapot.',
                },
            ],
        },
        'motion': {
            'allowed_penetration': 0.015,
            'z_rotation_samples': 8,
            'post_release_hover_m': 0.08,
            'preinsertion': {
                'use_preinsertion': True,
                'axis': '+z',
                'offset_m': 0.08,
                'position_range': np.array([0.03, 0.03, 0.02]),
                'position_samples': 50,
                'waypoint_allowance': 3,
                'allowed_penetration': 0.015,
            },
        },
        'digital_twin': {
            'sampling': {
                'n_samples': 500,
                'normal_position_range': np.array([0.02, 0.02, 0.02]),
                'normal_orientation_range': np.array([60.0, 60.0, 60.0]),
                'sim_steps': 500,
            },
            'validation_program': {
                'validation_mode': 'settling',
                'initial_max_pen': 0.01,
                'final_max_pen': 0.001,
                'required_contacts': ['active_passive'],
                'forbidden_contacts': ['active_table'],
                'xyz_relations': [{
                    'axis': 'z',
                    'relation': 'greater_than',
                    'subject': 'active',
                    'reference': 'passive',
                }],
                'allowed_position_drift_fraction': _world_drift_fractions(
                    0.03, 'yellow_cap'
                ),
                'allowed_orientation_drift_deg': 15.0,
                'reasoning': 'The cap must remain supported above the teapot without touching the table.',
            },
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (2.0, 2.0, 5), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (-2.0, -2.0, 5), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
    },
    'pour_water': {
        'task': {
            'objects': {
                'red_mug_big': {'source_pose': {'xyz': (0.0, 0.0, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'teapot': {'source_pose': {'xyz': (-0.1, 0.35, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
            },
            'default_stages': [
                {
                    'active': 'teapot',
                    'passive': 'red_mug_big',
                    'instruction': 'Pour water from the teapot into the red mug.',
                },
            ],
        },
        'motion': {
            'allowed_penetration': 0.003,
            'z_rotation_samples': 1,
            'post_release_hover_m': 0.08,
            'preinsertion': {
                'use_preinsertion': False,
                'axis': '+z',
                'offset_m': 0.1,
                'position_range': np.array([0.04, 0.04, 0.04]),
                'position_samples': 50,
                'waypoint_allowance': 3,
                'allowed_penetration': 0.003,
            },
        },
        'digital_twin': {
            'sampling': {
                'n_samples': 500,
                'normal_position_range': np.array([0.02, 0.02, 0.02]),
                'normal_orientation_range': np.array([60.0, 60.0, 60.0]),
                'sim_steps': 2,
            },
            'validation_program': {
                'validation_mode': 'geometric_only',
                'initial_max_pen': 0.1,
                'final_max_pen': 0.003,
                'required_contacts': [],
                'forbidden_contacts': ['active_table'],
                'xyz_relations': [{
                    'axis': 'z',
                    'relation': 'greater_than',
                    'subject': 'active',
                    'reference': 'passive',
                }],
                'allowed_position_drift_fraction': None,
                'allowed_orientation_drift_deg': None,
                'reasoning': 'A pouring pose is intentionally unsupported, but the teapot origin should be above the mug and clear of the table.',
            },
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -0.8, 1), 'up': (0.0, 0.0, 1.0), 'zoom': 1.4},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 1), 'up': (0.0, 0.0, 1.0), 'zoom': 1.4},
        ),
    },
    'mug_on_rack': {
        'task': {
            'objects': {
                # 'red_mug_big': {'source_pose': {'xyz': (0.0150, 0.4050, SURFACE_HEIGHT_M + 0.075), 'rpy_deg': (0.0, 77.0, 140.0)}},
                'red_mug_big': {'source_pose': {'xyz': (0.030, 0.2050, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'wooden_stick_toy': {'source_pose': {'xyz': (-0.05, 0.49, SURFACE_HEIGHT_M), 'rpy_deg': (0.0, 0.0, -90.0)}},
            },
            'default_stages': [
                {
                    'active': 'red_mug_big',
                    'passive': 'wooden_stick_toy',
                    'instruction': "Hang the red mug on the wooden rack's stick by its handle. The mug should be close to the rack's trunk.",
                },
            ],
        },
        'motion': {
            'allowed_penetration': 0.004,
            'z_rotation_samples': 1,
            'preinsertion': {
                'use_preinsertion': False,
                'axis': '+y',
                'offset_m': 0.12,
                'position_range': np.array([0.04, 0.04, 0.04]),
                'position_samples': 100,
                'waypoint_allowance': 3,
                'allowed_penetration': 0.01,
            },
        },
        'digital_twin': {
            'sampling': {
                'n_samples': 5000,
                'normal_position_range': np.array([0.12, 0.06, 0.06]),
                'normal_orientation_range': np.array([120.0, 120.0, 120.0]),
                'sim_steps': 500,
                'pos_bias': {'x': 'negative', 'y': None, 'z': 'positive'},
            },
            'validation_program': {
                'validation_mode': 'settling',
                'initial_max_pen': 0.015,
                'final_max_pen': 0.0001,
                'required_contacts': ['active_passive'],
                'forbidden_contacts': ['active_table'],
                # Intentionally replace the legacy absolute min_active_z floor
                # with the generic body-origin ordering relation.
                'xyz_relations': [{
                    'axis': 'z',
                    'relation': 'greater_than',
                    'subject': 'active',
                    'reference': 'passive',
                }],
                'allowed_position_drift_fraction': _world_drift_fractions(
                    0.1, 'red_mug_big'
                ),
                'allowed_orientation_drift_deg': 20.0,
                'reasoning': 'The mug must remain supported above the rack origin without touching the table.',
            },
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (6.0, -6.0, 6.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (6.0, 6.0, 6.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
    },
}

# Shared defaults removed from the individual benchmark dictionaries.
SAMPLING_DEFAULTS = {
    "max_candidate_images": 5,
    "position_variance_multipliers": {"low": 0.25, "medium": 1.0, "high": 2.0},
    "orientation_variance_multipliers": {"low": 0.25, "medium": 1.0, "high": 2.0},
    "pos_bias": {"x": None, "y": None, "z": None},
    "angular_kick_speed": 0.05,
}

PREINSERTION_DEFAULTS = {
    "use_preinsertion": False,
    "axis": "+z",
    "offset_m": 0.0,
    "position_range": np.zeros(3),
    "position_samples": 25,
    "waypoint_allowance": 0,
    "stop_short_waypoints": 0,
    "allowed_penetration": 0.003,
}


# Robot and planner defaults.
ARM_JOINT_NAMES = "arm1_joint1,arm1_joint2,arm1_joint3,arm1_joint4,arm1_joint5,arm1_joint6,arm1_joint7"
EEF_BODY_NAME = "arm1_box_gripper"
GRIPPER_ACTUATOR_SUFFIX = "gripper"
# The two finger slides start at q=0 with a 50 mm gap; increasing the tendon
# command closes them toward the center. Keep approach/release at the open end.
OPEN_CTRL_VALUE = 0.0
STAGE_INTERPOLATE_COUNT = 100
OMPL_SOLVE_TIME = 5.0
OMPL_INTERPOLATE_COUNT = 120
LIFT_HEIGHT_M = 0.10
DEFAULT_REALSENSE_SERIAL = "243622060451"
MAX_JOINT_VEL = 0.2


def object_prompt(object_id: str) -> str:
    return OBJECT_CATALOG.get(object_id, {}).get("prompt", object_id.replace("_", " "))


def _normalize_stage(stage: dict | tuple | list, *, candidate_objects: list[str]) -> dict:
    if isinstance(stage, dict):
        active = stage.get("active") or stage.get("active_object")
        passive = stage.get("passive") or stage.get("passive_object")
        reason = stage.get("reason", "")
        instruction = stage.get("instruction", "")
        raw_validation_program = stage.get("validation_program")
    else:
        active, passive = stage
        reason = ""
        instruction = ""
        raw_validation_program = None

    if active not in candidate_objects:
        raise ValueError(f"Stage active object {active!r} is not in candidate objects.")
    if passive not in candidate_objects:
        raise ValueError(f"Stage passive object {passive!r} is not in candidate objects.")
    if active == passive:
        raise ValueError(f"Stage active/passive objects must differ: {active!r}")

    normalized = {
        "active": active,
        "passive": passive,
        "active_object": active,
        "passive_object": passive,
        "reason": str(reason),
        "instruction": str(instruction),
    }
    if raw_validation_program is not None:
        normalized["validation_program"] = ValidationProgram.model_validate(
            raw_validation_program
        ).model_dump(mode="json")
    return normalized


def _source_pose(object_config: dict) -> dict | None:
    return object_config.get("source_pose") or object_config.get("pose")


def _merged_sampling(raw: dict) -> dict:
    sampling = copy.deepcopy(SAMPLING_DEFAULTS)
    supplied = copy.deepcopy(raw)
    sampling.update(supplied)
    sampling["position_variance_multipliers"] = {
        **SAMPLING_DEFAULTS["position_variance_multipliers"],
        **supplied.get("position_variance_multipliers", {}),
    }
    sampling["orientation_variance_multipliers"] = {
        **SAMPLING_DEFAULTS["orientation_variance_multipliers"],
        **supplied.get("orientation_variance_multipliers", {}),
    }
    sampling["pos_bias"] = {
        **SAMPLING_DEFAULTS["pos_bias"],
        **supplied.get("pos_bias", {}),
    }
    return sampling


def _merged_motion(raw: dict) -> dict:
    motion = copy.deepcopy(raw)
    motion.setdefault("allowed_penetration", 0.003)
    motion.setdefault("z_rotation_samples", 1)
    motion.setdefault("post_release_hover_m", 0.0)
    preinsertion = copy.deepcopy(PREINSERTION_DEFAULTS)
    preinsertion.update(copy.deepcopy(motion.get("preinsertion", {})))
    motion["preinsertion"] = preinsertion
    return motion


def derive_scene_fields(scene: dict) -> dict:
    """Derive mesh paths and MuJoCo names from the active/passive IDs."""
    scene = copy.deepcopy(scene)
    active_name = scene.get("active_object")
    passive_name = scene.get("passive_object")
    if not active_name or not passive_name:
        raise ValueError("Scene config requires active_object and passive_object.")

    try:
        active = copy.deepcopy(OBJECT_CATALOG[active_name])
        passive = copy.deepcopy(OBJECT_CATALOG[passive_name])
    except KeyError as exc:
        raise ValueError(f"Unknown object in scene: {exc.args[0]!r}") from exc

    active.update(scene.get("active_asset") or {})
    passive.update(scene.get("passive_asset") or {})
    scene.setdefault("active_mesh", active.get("mesh"))
    scene.setdefault("passive_mesh", passive.get("mesh"))
    scene.setdefault("mujoco_active_body", active_name)
    scene.setdefault("object_freejoint_name", f"{active_name}_free")
    scene.setdefault("goal_body_name", f"{active_name}_goal")
    scene.setdefault("goal_freejoint_name", f"{active_name}_goal_free")
    scene.setdefault(
        "contact_entities",
        {"active": active_name, "passive": passive_name, "table": "table"},
    )
    return scene


def _normalize_benchmark_config(
    raw_config: dict,
    *,
    benchmark: str,
    stage: dict | None = None,
    stage_index: int = 0,
    object_poses: dict | None = None,
) -> dict:
    config = copy.deepcopy(raw_config)
    task = config["task"]
    objects = task.get("objects") or {}
    object_configs = objects if isinstance(objects, dict) else {obj: {} for obj in objects}
    candidate_objects = list(object_configs)
    if not candidate_objects:
        raise ValueError(f"Benchmark {benchmark!r} has no task objects.")

    unknown = [obj for obj in candidate_objects if obj not in OBJECT_CATALOG]
    if unknown:
        raise ValueError(f"Unknown task object ids: {unknown}")

    object_configs = {
        obj: copy.deepcopy(object_configs.get(obj, {})) for obj in candidate_objects
    }
    default_stages = [
        _normalize_stage(item, candidate_objects=candidate_objects)
        for item in task.get("default_stages", [])
    ]
    if stage is not None:
        selected_stage = _normalize_stage(stage, candidate_objects=candidate_objects)
        # Runtime stage resolution passes a fresh stage dict. Recover a configured
        # per-stage fallback when the resolved active/passive pair identifies one.
        matching_defaults = [
            item
            for item in default_stages
            if item["active"] == selected_stage["active"]
            and item["passive"] == selected_stage["passive"]
        ]
        if (
            "validation_program" not in selected_stage
            and len(matching_defaults) == 1
            and "validation_program" in matching_defaults[0]
        ):
            selected_stage["validation_program"] = copy.deepcopy(
                matching_defaults[0]["validation_program"]
            )
    elif default_stages:
        if not 0 <= stage_index < len(default_stages):
            raise IndexError(
                f"Stage index {stage_index} is outside benchmark {benchmark!r} "
                f"with {len(default_stages)} configured stage(s)."
            )
        selected_stage = default_stages[stage_index]
    else:
        raise ValueError(
            f"Benchmark {benchmark!r} requires default_stages or an explicit stage."
        )

    scene = config.setdefault("scene", {})
    motion = _merged_motion(config.get("motion", {}))
    digital_twin = copy.deepcopy(config.get("digital_twin", {}))
    sampling = _merged_sampling(digital_twin.get("sampling", {}))
    raw_validation_program = selected_stage.get(
        "validation_program", digital_twin.get("validation_program")
    )
    if raw_validation_program is None:
        raise ValueError(
            f"Benchmark {benchmark!r} requires a complete validation program."
        )
    validation_program = ValidationProgram.model_validate(raw_validation_program)
    digital_twin["sampling"] = sampling
    digital_twin["validation_program"] = validation_program.model_dump(mode="json")
    config["motion"] = motion
    config["digital_twin"] = digital_twin

    active = selected_stage["active"]
    passive = selected_stage["passive"]
    active_config = object_configs[active]
    passive_config = object_configs[passive]
    default_object_poses = {
        obj: pose
        for obj, object_config in object_configs.items()
        if (pose := _source_pose(object_config)) is not None
    }

    scene["active_object"] = active
    scene["passive_object"] = passive
    scene["candidate_objects"] = list(candidate_objects)
    scene["object_poses"] = copy.deepcopy(object_poses or {})
    scene["default_object_poses"] = default_object_poses
    scene["active_source_pose"] = _source_pose(active_config)
    scene["passive_source_pose"] = _source_pose(passive_config)

    motion["grasps"] = copy.deepcopy(OBJECT_CATALOG[active].get("grasps", []))
    task["candidate_objects"] = list(candidate_objects)
    task["object_prompts"] = {
        obj: object_prompt(obj) for obj in candidate_objects
    }
    task["default_stages"] = default_stages
    task["stage"] = selected_stage
    task["active_prompt"] = object_prompt(active)
    task["passive_prompt"] = object_prompt(passive)
    task["use_vlm_stage_resolution"] = bool(
        task.get("use_vlm_stage_resolution", False)
    )
    task["skip_sampling"] = bool(task.get("skip_sampling", False))

    config["scene"] = derive_scene_fields(scene)
    return config


def get_benchmark_config(
    benchmark: str | None = None,
    *,
    stage: dict | None = None,
    stage_index: int = 0,
    object_poses: dict | None = None,
) -> dict:
    name = benchmark or BENCHMARK
    try:
        return _normalize_benchmark_config(
            BENCHMARKS[name],
            benchmark=name,
            stage=stage,
            stage_index=stage_index,
            object_poses=object_poses,
        )
    except KeyError as exc:
        choices = ", ".join(BENCHMARKS)
        raise ValueError(f"Unknown benchmark {name!r}. Choose from: {choices}") from exc


@dataclass(frozen=True)
class PipelineConfig:
    benchmark: str
    instruction: str
    active_prompt: str | None
    passive_prompt: str | None
    use_vlm_stage_resolution: bool
    active_mesh: str
    passive_mesh: str
    mujoco_active_body: str
    normal_position_range: np.ndarray
    normal_orientation_range_deg: np.ndarray
    n_samples: int
    max_candidate_images: int
    object_freejoint_name: str
    z_rotation_samples: int
    allowed_penetration: float
    grasp_candidates: tuple[dict, ...]
    use_preinsertion: bool
    preinsertion_axis: str
    preinsertion_offset_m: float
    preinsertion_position_range: np.ndarray
    preinsertion_position_samples: int
    preinsertion_waypoint_allowance: int
    preinsertion_stop_short_waypoints: int
    preinsertion_allowed_penetration: float
    allowed_contact_pair: tuple[str, ...]
    post_release_hover_m: float
    camera_type: str
    camera_serial: str | None
    axis_reasoning: bool
    debug: bool
    skip_sampling: bool
    skip_settlement: bool
    board_x_mj: float
    board_y_mj: float
    stage_index: int = 0
    stage: dict | None = None
    default_stages: tuple[dict, ...] = ()
    candidate_objects: tuple[str, ...] = ()
    object_prompts: dict[str, str] | None = None
    object_poses: dict[str, dict] | None = None
    stage_count: int = 1


def _runtime_config_from_normalized(
    config: dict,
    *,
    benchmark: str,
    camera_type: str,
    camera_serial: str | None,
    axis_reasoning: bool,
    debug: bool,
    board_x_mj: float,
    board_y_mj: float,
) -> PipelineConfig:
    task, scene = config["task"], config["scene"]
    motion = config["motion"]
    sampling = config["digital_twin"]["sampling"]
    grasps = tuple(motion["grasps"])
    if not grasps:
        raise ValueError(f"Benchmark {benchmark!r} active object has no grasp candidates.")
    pre = motion["preinsertion"]

    return PipelineConfig(
        benchmark=benchmark,
        instruction=task["stage"]["instruction"],
        active_prompt=task["active_prompt"],
        passive_prompt=task["passive_prompt"],
        use_vlm_stage_resolution=task["use_vlm_stage_resolution"],
        active_mesh=scene["active_mesh"],
        passive_mesh=scene["passive_mesh"],
        mujoco_active_body=scene["mujoco_active_body"],
        normal_position_range=np.asarray(sampling["normal_position_range"], dtype=float),
        normal_orientation_range_deg=np.asarray(sampling["normal_orientation_range"], dtype=float),
        n_samples=int(sampling["n_samples"]),
        max_candidate_images=int(sampling["max_candidate_images"]),
        object_freejoint_name=scene["object_freejoint_name"],
        z_rotation_samples=int(motion["z_rotation_samples"]),
        allowed_penetration=float(motion["allowed_penetration"]),
        grasp_candidates=grasps,
        use_preinsertion=bool(pre["use_preinsertion"]),
        preinsertion_axis=str(pre["axis"]),
        preinsertion_offset_m=float(pre["offset_m"]),
        preinsertion_position_range=np.asarray(pre["position_range"], dtype=float),
        preinsertion_position_samples=int(pre["position_samples"]),
        preinsertion_waypoint_allowance=int(pre["waypoint_allowance"]),
        preinsertion_stop_short_waypoints=int(pre["stop_short_waypoints"]),
        preinsertion_allowed_penetration=float(pre["allowed_penetration"]),
        allowed_contact_pair=(
            f"arm1_:{scene['mujoco_active_body']}",
            f"gripper:{scene['mujoco_active_body']}",
            *scene.get("extra_allowed_contact_pair", ()),
        ),
        post_release_hover_m=float(motion["post_release_hover_m"]),
        camera_type=camera_type,
        camera_serial=camera_serial,
        axis_reasoning=bool(axis_reasoning),
        debug=bool(debug),
        skip_sampling=task["skip_sampling"],
        skip_settlement=bool(motion.get("skip_settlement", False)),
        board_x_mj=float(board_x_mj),
        board_y_mj=float(board_y_mj),
        stage=task["stage"],
        default_stages=tuple(task["default_stages"]),
        candidate_objects=tuple(task["candidate_objects"]),
        object_prompts=task["object_prompts"],
        object_poses=copy.deepcopy(scene["object_poses"]),
        stage_count=max(1, len(task["default_stages"])),
    )


def build_pipeline_config(
    benchmark: str,
    *,
    camera_type: str = CAMERA_TYPE,
    camera_serial: str | None = None,
    axis_reasoning: bool = True,
    debug: bool = False,
    board_x_mj: float = -0.12,
    board_y_mj: float = 0.0,
) -> PipelineConfig:
    config = get_benchmark_config(benchmark)
    serial = camera_serial or (
        DEFAULT_REALSENSE_SERIAL if camera_type == "realsense" else None
    )
    return _runtime_config_from_normalized(
        config,
        benchmark=benchmark,
        camera_type=camera_type,
        camera_serial=serial,
        axis_reasoning=axis_reasoning,
        debug=debug,
        board_x_mj=board_x_mj,
        board_y_mj=board_y_mj,
    )


def stage_dict(stage) -> dict:
    return {
        "active": stage.active_object,
        "passive": stage.passive_object,
        "reason": stage.reason,
        "instruction": stage.instruction,
    }


def pose_dict(T: np.ndarray) -> dict:
    return {
        "xyz": tuple(float(value) for value in T[:3, 3]),
        "quat": tuple(float(value) for value in mat_to_quat_wxyz(T[:3, :3])),
    }


def stage_config(args: PipelineConfig) -> dict:
    return get_benchmark_config(
        args.benchmark,
        stage=args.stage,
        stage_index=args.stage_index,
        object_poses=args.object_poses,
    )


def for_stage(
    args: PipelineConfig,
    stage,
    stage_index: int,
    object_poses: dict,
    stage_count: int,
) -> PipelineConfig:
    config = get_benchmark_config(
        args.benchmark,
        stage=stage_dict(stage),
        stage_index=stage_index,
        object_poses=object_poses,
    )
    stage_args = _runtime_config_from_normalized(
        config,
        benchmark=args.benchmark,
        camera_type=args.camera_type,
        camera_serial=args.camera_serial,
        axis_reasoning=args.axis_reasoning,
        debug=args.debug,
        board_x_mj=args.board_x_mj,
        board_y_mj=args.board_y_mj,
    )
    return replace(
        stage_args,
        use_vlm_stage_resolution=False,
        stage_index=stage_index,
        object_poses=copy.deepcopy(object_poses),
        stage_count=stage_count,
    )
