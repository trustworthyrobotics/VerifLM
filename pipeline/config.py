from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np

from pipeline.core import mat_to_quat_wxyz


# Object descriptions, assets, physics properties, and configured grasps.
OBJECT_CATALOG = {
    'wooden_rack': {
        'prompt': 'wooden rack',
        'mesh': 'models/meshes/wooden_rack/wooden_rack.obj',
        'asset_xml': 'meshes/wooden_rack_remeshed/wooden_rack_asset.xml',
        'body_xml': 'meshes/wooden_rack_remeshed/wooden_rack_body.xml',
        'geom_friction': '0.05 0.001 0.0001',
        'geom_condim': '4',
    },
    'wooden_stick_toy': {
        'prompt': 'wooden stick toy',
        'mesh': 'models/meshes/wooden_stick_toy/wooden_stick_toy.obj',
        'asset_xml': 'meshes/wooden_stick_toy_remeshed/wooden_stick_toy_asset.xml',
        'body_xml': 'meshes/wooden_stick_toy_remeshed/wooden_stick_toy_body.xml',
        'geom_friction': '0.8 0.02 0.001',
    },
    'red_mug_small': {
        'prompt': 'red mug small',
        'mesh': 'models/meshes/red_mug_small/red_mug_small.obj',
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
    "step_8": "openai",   # sampling-axis reasoning
    "step_10": "gemini",  # candidate selection
}


# Benchmark definitions contain only values that differ by task.
BENCHMARKS = {
    'mug_on_plate': {
        'scene': {'surface_height': 0.7893, 'generated_xml_path': 'models/generated_xmls/red_mug_on_plate.xml'},
        'task': {
            'objects': {
                'red_mug_big': {'source_pose': {'xyz': (-0.2, 0.4, 0.83), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'blue_plate_big': {'source_pose': {'xyz': (0, 0, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
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
        'sampling': {
            'n_samples': 1000,
            'normal_position_range': np.array([0.03, 0.03, 0.03]),
            'normal_orientation_range': np.array([0.17453292519943295, 0.17453292519943295, 0.17453292519943295]),
            'initial_max_pen': 0.01,
            'final_max_pen': 0.001,
            'sim_steps': 500,
            'max_z_change': 0.05,
            'max_ori_change': 0.06981317007977318,
            'use_axis_reasoning': True,
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (-1, -1, 2), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (1, 1, 2), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
        ),
    },
    'mugs_on_plates': {
        'scene': {'surface_height': 0.7893},
        'task': {
            'objects': {
                'red_mug_big': {'source_pose': {'xyz': (-0.2, 0.4, 0.83), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'red_plate_big': {'source_pose': {'xyz': (0.0, 0.0, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'blue_mug_big': {'source_pose': {'xyz': (0.2, 0.4, 0.83), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'blue_plate_big': {'source_pose': {'xyz': (0.4, 0.0, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
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
        'sampling': {
            'n_samples': 1000,
            'normal_position_range': np.array([0.03, 0.03, 0.03]),
            'normal_orientation_range': np.array([0.17453292519943295, 0.17453292519943295, 0.17453292519943295]),
            'initial_max_pen': 0.01,
            'final_max_pen': 0.001,
            'sim_steps': 500,
            'max_z_change': 0.05,
            'max_ori_change': 0.06981317007977318,
            'use_axis_reasoning': True,
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (-1, -1, 2), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (1, 1, 2), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
        ),
    },
    'stack_cups': {
        'scene': {'surface_height': 0.7893, 'generated_xml_path': 'models/generated_xmls/stacked_cups.xml'},
        'task': {
            'objects': {
                'red_cup': {'source_pose': {'xyz': (-0.18, 0.22, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'green_cup': {'source_pose': {'xyz': (0.0, 0.22, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'blue_cup': {'source_pose': {'xyz': (0.18, 0.22, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
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
        'sampling': {
            'n_samples': 600,
            'normal_position_range': np.array([0.03, 0.03, 0.03]),
            'normal_orientation_range': np.array([0.3490658503988659, 0.3490658503988659, 0.3490658503988659]),
            'initial_max_pen': 0.08,
            'final_max_pen': 0.001,
            'sim_steps': 500,
            'max_z_change': 0.1,
            'max_ori_change': 0.5235987755982988,
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -1.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
    },
    'plate_on_rack': {
        'scene': {'generated_xml_path': 'models/generated_xmls/plate_on_rack.xml', 'surface_height': 0.7893},
        'task': {
            'objects': {
                'green_plate_small': {'source_pose': {'xyz': (-0.015, 0.285, 0.8893), 'rpy_deg': (-90.88, -9.96, 95.0)}},
                'wooden_rack': {'source_pose': {'xyz': (-0.04, 0.49, 0.7893), 'rpy_deg': (0.0, 0.0, 90.0)}},
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
        'sampling': {
            'n_samples': 1000,
            'normal_position_range': np.array([0.03, 0.005, 0.02]),
            'normal_orientation_range': np.array([1.5707963267948966, 0.5235987755982988, 0.17453292519943295]),
            'initial_max_pen': 0.2,
            'final_max_pen': 0.001,
            'sim_steps': 600,
            'max_z_change': 0.08,
            'max_ori_change': 0.13962634015954636,
            'pos_bias': {'x': 'negative', 'y': None, 'z': 'negative'},
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.4, 0.0, 0.8), 'up': (0.0, 0.0, 1.0), 'zoom': 1.0},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -0.4, 0.4), 'up': (0.0, 0.0, 1.0), 'zoom': 1.0},
        ),
    },
    'plates_on_rack': {
        'scene': {'surface_height': 0.7893},
        'task': {
            'objects': {
                'green_plate_small': {'source_pose': {'xyz': (-0.2, 0.75, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'yellow_plate_small': {'source_pose': {'xyz': (0.05, 0.75, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'wooden_rack': {'source_pose': {'xyz': (-0.12, 0.6, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
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
        'sampling': {
            'n_samples': 1000,
            'normal_position_range': np.array([0.02, 0.02, 0.02]),
            'normal_orientation_range': np.array([1.0471975511965976, 1.0471975511965976, 1.0471975511965976]),
            'initial_max_pen': 0.1,
            'final_max_pen': 0.001,
            'sim_steps': 500,
            'max_z_change': 0.08,
            'max_ori_change': 0.13962634015954636,
            'use_axis_reasoning': True,
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 1.5), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -0.8, 0.8), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
    },
    'stack_plate': {
        'task': {
            'objects': {
                'yellow_plate_small': {'source_pose': {'xyz': (-0.1, 0.65, 0.8593), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'green_plate_small': {'source_pose': {'xyz': (0.1567, 0.3621, 1.0202), 'rpy_deg': (0.0, 90.0, 0.0)}},
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
        'sampling': {
            'n_samples': 1000,
            'normal_position_range': np.array([0.02, 0.02, 0.04]),
            'normal_orientation_range': np.array([0.3490658503988659, 0.3490658503988659, 0.3490658503988659]),
            'initial_max_pen': 0.08,
            'final_max_pen': 0.001,
            'sim_steps': 500,
            'max_z_change': 0.05,
            'max_ori_change': 0.13962634015954636,
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -1.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
        'scene': {},
    },
    'stack_plates': {
        'scene': {'surface_height': 0.7893},
        'task': {
            'objects': {
                'green_plate_small': {'source_pose': {'xyz': (-0.2, 0.65, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'yellow_plate_small': {'source_pose': {'xyz': (0.0, 0.65, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'pink_plate_small': {'source_pose': {'xyz': (0.2, 0.65, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
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
        'sampling': {
            'n_samples': 1000,
            'normal_position_range': np.array([0.03, 0.03, 0.02]),
            'normal_orientation_range': np.array([0.3490658503988659, 0.3490658503988659, 0.3490658503988659]),
            'initial_max_pen': 0.08,
            'final_max_pen': 0.001,
            'sim_steps': 500,
            'max_z_change': 0.05,
            'max_ori_change': 0.13962634015954636,
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -1.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
    },
    'cap_on_teapot': {
        'scene': {
            'surface_height': 0.7893,
            'extra_allowed_contact_pair': ('arm1_:teapot',),
            'generated_xml_path': 'models/generated_xmls/cap_on_teapot.xml',
        },
        'task': {
            'objects': {
                'yellow_cap': {'source_pose': {'xyz': (0.0, 0.34, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'teapot': {'source_pose': {'xyz': (-0.12, 0.2, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
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
        'sampling': {
            'n_samples': 500,
            'normal_position_range': np.array([0.02, 0.02, 0.02]),
            'normal_orientation_range': np.array([0.3490658503988659, 0.3490658503988659, 0.3490658503988659]),
            'initial_max_pen': 0.01,
            'final_max_pen': 0.001,
            'sim_steps': 500,
            'max_z_change': 0.03,
            'max_ori_change': 0.2617993877991494,
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (2.0, 2.0, 5), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (-2.0, -2.0, 5), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
    },
    'pour_water': {
        'scene': {'surface_height': 0.7893, 'generated_xml_path': 'models/generated_xmls/pour_water.xml'},
        'task': {
            'objects': {
                'red_mug_big': {'source_pose': {'xyz': (0.0, 0.0, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
                'teapot': {'source_pose': {'xyz': (-0.1, 0.35, 0.7893), 'rpy_deg': (0.0, 0.0, 0.0)}},
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
        'sampling': {
            'n_samples': 500,
            'normal_position_range': np.array([0.02, 0.02, 0.02]),
            'normal_orientation_range': np.array([0.3490658503988659, 0.3490658503988659, 0.3490658503988659]),
            'initial_max_pen': 0.1,
            'final_max_pen': 0.003,
            'sim_steps': 2,
            'max_z_change': 0.04,
            'max_ori_change': 0.6108652381980153,
            'required_contacts': [],
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -0.8, 1), 'up': (0.0, 0.0, 1.0), 'zoom': 1.4},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 1), 'up': (0.0, 0.0, 1.0), 'zoom': 1.4},
        ),
    },
    'mug_on_rack': {
        'scene': {'surface_height': 0.7893, 'generated_xml_path': 'models/generated_xmls/red_mug_on_rack.xml'},
        'task': {
            'objects': {
                'red_mug_big': {'source_pose': {'xyz': (-0.12, 0.275, 0.7893), 'rpy_deg': (0.0, 0.0, 140.0)}},
                'wooden_stick_toy': {'source_pose': {'xyz': (-0.05, 0.49, 0.7893), 'rpy_deg': (0.0, 0.0, -90.0)}},
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
                'use_preinsertion': True,
                'axis': '+y',
                'offset_m': 0.12,
                'position_range': np.array([0.04, 0.04, 0.04]),
                'position_samples': 100,
                'waypoint_allowance': 3,
                'allowed_penetration': 0.01,
            },
        },
        'sampling': {
            'n_samples': 5000,
            'normal_position_range': np.array([0.04, 0.05, 0.04]),
            'normal_orientation_range': np.array([2.0943951023931953, 1.0471975511965976, 2.0943951023931953]),
            'initial_max_pen': 0.015,
            'final_max_pen': 0.0001,
            'sim_steps': 500,
            'max_z_change': 0.1,
            'max_ori_change': 3.141592653589793,
            'min_active_z': 0.85,
            'pos_bias': {'x': 'positive', 'y': None, 'z': 'positive'},
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
    "use_axis_reasoning": False,
    "pos_bias": {"x": None, "y": None, "z": None},
    "required_contacts": [("active", "passive")],
    "forbidden_contacts": [("active", "table")],
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
OPEN_CTRL_VALUE = 0.05
STAGE_INTERPOLATE_COUNT = 45
OMPL_SOLVE_TIME = 5.0
OMPL_INTERPOLATE_COUNT = 120
LIFT_HEIGHT_M = 0.10
DEFAULT_REALSENSE_SERIAL = "243622060451"


def object_prompt(object_id: str) -> str:
    return OBJECT_CATALOG.get(object_id, {}).get("prompt", object_id.replace("_", " "))


def _normalize_stage(stage: dict | tuple | list, *, candidate_objects: list[str]) -> dict:
    if isinstance(stage, dict):
        active = stage.get("active") or stage.get("active_object")
        passive = stage.get("passive") or stage.get("passive_object")
        reason = stage.get("reason", "")
        instruction = stage.get("instruction", "")
    else:
        active, passive = stage
        reason = ""
        instruction = ""

    if active not in candidate_objects:
        raise ValueError(f"Stage active object {active!r} is not in candidate objects.")
    if passive not in candidate_objects:
        raise ValueError(f"Stage passive object {passive!r} is not in candidate objects.")
    if active == passive:
        raise ValueError(f"Stage active/passive objects must differ: {active!r}")

    return {
        "active": active,
        "passive": passive,
        "active_object": active,
        "passive_object": passive,
        "reason": str(reason),
        "instruction": str(instruction),
    }


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
    sampling = _merged_sampling(config.get("sampling", {}))
    config["motion"] = motion
    config["sampling"] = sampling

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
    if "surface_height" in scene:
        scene["surface_height"] = float(scene["surface_height"])
    scene["object_poses"] = copy.deepcopy(object_poses or {})
    scene["default_object_poses"] = default_object_poses
    scene["active_source_pose"] = _source_pose(active_config)
    scene["passive_source_pose"] = _source_pose(passive_config)
    if len(default_stages) > 1 or stage_index > 0:
        scene["generated_xml_path"] = (
            f"models/generated_xmls/{benchmark}_stage_{stage_index + 1:02d}.xml"
        )
    else:
        scene.setdefault("generated_xml_path", f"models/generated_xmls/{benchmark}.xml")

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
    motion, sampling = config["motion"], config["sampling"]
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
        normal_orientation_range_deg=np.rad2deg(sampling["normal_orientation_range"]),
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
        axis_reasoning=bool(axis_reasoning or sampling["use_axis_reasoning"]),
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
    axis_reasoning: bool = False,
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
