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

BENCHMARK_CONFIGS = {
    'mug_on_plate': {
        'active': 'red_mug_big',
        'passive': 'blue_plate_big',
        'instruction': 'Place the red mug on the blue plate.',
        'default_validation_program': {
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
            'allowed_position_drift_fraction': 0.5,
            'allowed_orientation_drift_deg': 20,
        },
        'render_config': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (-1, -1, 2), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (1, 1, 2), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
        ),
    },
    'mug_on_rack': {
        'active': 'red_mug_big',
        'passive': 'wooden_stick_toy',
        'instruction': 'Hang the red mug on the wooden rack by its handle',
        'default_validation_program': {
            'validation_mode': 'settling',
            'initial_max_pen': 0.015,
            'final_max_pen': 0.0001,
            'required_contacts': ['active_passive'],
            'forbidden_contacts': ['active_table'],
            'xyz_relations': [{
                'axis': 'z',
                'relation': 'greater_than',
                'subject': 'active',
                'reference': 'passive',
            }],
            'allowed_position_drift_fraction': 0.5,
            'allowed_orientation_drift_deg': 20.0,
        },
        'render_config': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (6.0, -6.0, 6.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (6.0, 6.0, 6.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
    },
    'plate_on_rack': {
        'active': 'green_plate_small',
        'passive': 'wooden_rack',
        'instruction': 'Insert the green plate into a pair of opposing slots.',
        'default_validation_program': {
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
            'allowed_position_drift_fraction': 0.5,
            'allowed_orientation_drift_deg': 22.0,
        },
        "render_config": (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.4, 0.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 0.9},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -0.4, 0.4), 'up': (0.0, 0.0, 1.0), 'zoom': 0.8},
        ),
    },
    'cup_on_rack': {},
    'pour_tea': {},
    'lid_on_teapot': {},

    # Multi-stage tasks
    'stack_cups': {
        'active': ('red_cup', 'green_cup'),
        'passive': ('blue_cup', 'red_cup'),
        'instruction': ('Place the red cup inside the blue cup.', 'Place the green cup inside the red cup.'),
        'default_validation_program': {
            'validation_mode': 'settling',
            'initial_max_pen': 0.2,
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
            'allowed_position_drift_fraction': 0.5,
            'allowed_orientation_drift_deg': 20.0,
        },
        'views': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -1.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
    },
    'plates_on_rack': {},

    # Non-prehensile tasks
    'plate_on_rack_nonprehensile': {},
    'book_on_stand_nonprehensile': {},
    'order_letters': {},
}
