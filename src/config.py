OBJECT_CATALOG = {
    "wooden_rack": {
        "prompt": "wooden rack", "mesh": "models/meshes/wooden_rack/wooden_rack.obj",
        "dimensions": (0.2000, 0.1003, 0.0551),
        "mjcf": {"type": "mesh", "asset_xml": "models/meshes/wooden_rack_remeshed/wooden_rack_asset.xml",
                 "body_xml": "models/meshes/wooden_rack_remeshed/wooden_rack_body.xml"},
        "physics": {"mass": 0.4, "friction": 0.3},
    },
    "wooden_stick_toy": {
        "prompt": "wooden stick toy", "mesh": "models/meshes/wooden_stick_toy/wooden_stick_toy.obj",
        "dimensions": (0.1429, 0.1005, 0.2067),
        "mjcf": {"type": "mesh", "asset_xml": "models/meshes/wooden_stick_toy_remeshed/wooden_stick_toy_asset.xml",
                 "body_xml": "models/meshes/wooden_stick_toy_remeshed/wooden_stick_toy_body.xml"},
        "physics": {"mass": 0.3, "friction": 0.3},
    },
    "red_mug_small": {
        "prompt": "red mug small", "mesh": "models/meshes/red_mug_small/red_mug_small.obj",
        "dimensions": (0.0564, 0.0729, 0.0567),
        "mjcf": {"type": "mesh", "asset_xml": "models/meshes/red_mug_small_remeshed/red_mug_small_asset.xml",
                 "body_xml": "models/meshes/red_mug_small_remeshed/red_mug_small_body.xml"},
        "physics": {"mass": 0.12, "friction": 0.8},
        "grasps": [
            {"offset_xyz": (0.0, 0.0, 0.22165), "offset_rpy_deg": (180.0, 0.0, 0.0),
             "hover_offset_xyz": (0, 0, 0.1), "z_rotation_samples": 1},
        ],
    },
    "red_mug_big": {
        "prompt": "red mug", "mesh": "models/meshes/red_mug_big/red_mug_big.obj",
        "dimensions": (0.0879, 0.1108, 0.0876),
        "mjcf": {"type": "mesh", "asset_xml": "models/meshes/red_mug_big_remeshed/red_mug_big_asset.xml",
                 "body_xml": "models/meshes/red_mug_big_remeshed/red_mug_big_body.xml"},
        "physics": {"mass": 0.2, "friction": 0.1},
        "grasps": [
            {"offset_xyz": (-0.041, 0, 0.1262), "offset_rpy_deg": (180, 0, 0), "hover_offset_xyz": (0, 0, 0.08)},
            {"offset_xyz": (0.041, 0, 0.1262), "offset_rpy_deg": (180, 0, 0), "hover_offset_xyz": (0, 0, 0.08)},
            {"offset_xyz": (0, -0.041, 0.1262), "offset_rpy_deg": (180, 0, -90), "hover_offset_xyz": (0, 0, 0.08)},
            {"offset_xyz": (0, 0.041, 0.1262), "offset_rpy_deg": (180, 0, -90), "hover_offset_xyz": (0, 0, 0.08)},
            {"offset_xyz": (0, -0.041, 0.1262), "offset_rpy_deg": (180, 0, 90), "hover_offset_xyz": (0, 0, 0.08)},
            {"offset_xyz": (0, 0.041, 0.1262), "offset_rpy_deg": (180, 0, 90), "hover_offset_xyz": (0, 0, 0.08)},
            {"offset_xyz": (-0.0375, 0.0375, -0.13), "offset_rpy_deg": (0, 0, 45), "hover_offset_xyz": (0, 0, -0.02)},
            {"offset_xyz": (-0.0375, 0.0375, -0.13), "offset_rpy_deg": (0, 0, -135), "hover_offset_xyz": (0, 0, -0.02)},
            {"offset_xyz": (-0.07, 0.08, -0.13), "offset_rpy_deg": (30, 0, 45), "hover_offset_xyz": (0, 0, -0.02)},
            {"offset_xyz": (-0.07, 0.08, -0.13), "offset_rpy_deg": (-30, 0, -135), "hover_offset_xyz": (0, 0, -0.02)},
            {"offset_xyz": (-0.039, 0.039, 0.12), "offset_rpy_deg": (180, 0, 45), "hover_offset_xyz": (0, 0, 0.08)},
            {"offset_xyz": (-0.039, 0.039, 0.12), "offset_rpy_deg": (180, 0, -135), "hover_offset_xyz": (0, 0, 0.08)},
        ],
    },
    "blue_mug_big": {
        "prompt": "blue mug", "mesh": "models/meshes/blue_mug_big/blue_mug_big.obj",
        "dimensions": (0.0849, 0.1174, 0.0918),
        "mjcf": {"type": "mesh", "asset_xml": "models/meshes/blue_mug_big_remeshed/blue_mug_big_asset.xml",
                 "body_xml": "models/meshes/blue_mug_big_remeshed/blue_mug_big_body.xml"},
        "physics": {"mass": 0.2, "friction": 0.8},
        "grasps": [
            {"offset_xyz": (0, -0.037, 0.0841), "offset_rpy_deg": (180, 0, 90), "hover_offset_xyz": (0, 0, 0.08)},
            {"offset_xyz": (0, 0.037, 0.0641), "offset_rpy_deg": (180, 0, 90), "hover_offset_xyz": (0, 0, 0.08)},
            {"offset_xyz": (-0.037, 0, 0.0641), "offset_rpy_deg": (180, 0, 0), "hover_offset_xyz": (0, 0, 0.08)},
            {"offset_xyz": (0.037, 0, 0.0641), "offset_rpy_deg": (180, 0, 0), "hover_offset_xyz": (0, 0, 0.08)},
        ],
    },
    "green_plate_small": {
        "prompt": "green plate", "mesh": "models/meshes/green_plate_small/green_plate_small.obj",
        "dimensions": (0.1071, 0.1059, 0.0176),
        "mjcf": {"type": "primitive", "primitive": "small_dish", "z_offset": -0.0088},
        "physics": {"mass": 0.25, "friction": 0.3},
        "grasps": [
            {"offset_xyz": (0, 0.145, 0.0), "offset_rpy_deg": (90, -90, 0),
             "hover_offset_xyz": (0, 0.1, 0), "z_rotation_samples": 12},
        ],
    },
    "pink_plate_small": {
        "prompt": "pink plate", "mesh": "models/meshes/pink_plate_small/pink_plate_small.obj",
        "dimensions": (0.1076, 0.1078, 0.0180),
        "mjcf": {"type": "primitive", "primitive": "small_dish", "z_offset": -0.009},
        "physics": {"mass": 0.25, "friction": 0.3},
        "grasps": [
            {"offset_xyz": (0, 0.155, 0.0015), "offset_rpy_deg": (90, -90, 0),
             "hover_offset_xyz": (0, 0.1, 0), "z_rotation_samples": 12},
        ],
    },
    "yellow_plate_small": {
        "prompt": "yellow plate", "mesh": "models/meshes/yellow_plate_small/yellow_plate_small.obj",
        "dimensions": (0.1069, 0.1071, 0.0145),
        "mjcf": {"type": "primitive", "primitive": "small_dish", "z_offset": -0.00725},
        "physics": {"mass": 0.25, "friction": 0.3},
        "grasps": [
            {"offset_xyz": (0, 0.155, 0.0015), "offset_rpy_deg": (90, -90, 0),
             "hover_offset_xyz": (0, 0.1, 0), "z_rotation_samples": 12},
        ],
    },
    "red_plate_big": {
        "prompt": "red plate", "mesh": "models/meshes/red_plate_big/red_plate_big.obj",
        "dimensions": (0.1208, 0.1248, 0.0145),
        "mjcf": {"type": "primitive", "primitive": "big_dish", "z_offset": -0.00725},
        "physics": {"mass": 0.25, "friction": 0.3},
        "grasps": [
            {"offset_xyz": (0, 0.14, -0.00725), "offset_rpy_deg": (90, 90, 0),
             "hover_offset_xyz": (0, 0.1, 0), "z_rotation_samples": 12},
        ],
    },
    "blue_plate_big": {
        "prompt": "blue plate", "mesh": "models/meshes/blue_plate_big/blue_plate_big.obj",
        "dimensions": (0.1245, 0.1226, 0.0156),
        "mjcf": {"type": "primitive", "primitive": "big_dish", "z_offset": -0.0078},
        "physics": {"mass": 0.25, "friction": 0.3},
        "grasps": [
            {"offset_xyz": (0, 0.14, -0.0078), "offset_rpy_deg": (90, 90, 0),
             "hover_offset_xyz": (0, 0.1, 0), "z_rotation_samples": 12},
        ],
    },
    "teapot": {
        "prompt": "teapot", "mesh": "models/meshes/teapot/teapot.obj",
        "dimensions": (0.1676, 0.0919, 0.0961),
        "mjcf": {"type": "mesh", "asset_xml": "models/meshes/teapot_remeshed/teapot_asset.xml",
                 "body_xml": "models/meshes/teapot_remeshed/teapot_body.xml"},
        "physics": {"mass": 0.5, "friction": 0.15},
        "grasps": [
            {"offset_xyz": (0.075, 0, 0.12195), "offset_rpy_deg": (180, 0, -90), "hover_offset_xyz": (0, 0, 0.03)},
            {"offset_xyz": (0.075, 0, 0.12195), "offset_rpy_deg": (180, 0, 90), "hover_offset_xyz": (0, 0, 0.03)},
        ],
    },
    "yellow_cap": {
        "prompt": "yellow cap", "mesh": "models/meshes/yellow_cap/yellow_cap.obj",
        "dimensions": (0.0743, 0.0757, 0.0278),
        "mjcf": {"type": "mesh", "asset_xml": "models/meshes/yellow_cap_remeshed/yellow_cap_asset.xml",
                 "body_xml": "models/meshes/yellow_cap_remeshed/yellow_cap_body.xml"},
        "physics": {"mass": 0.03, "friction": 0.08},
        "grasps": [
            {"offset_xyz": (0, 0, 0.1191), "offset_rpy_deg": (180, 0, 0),
             "hover_offset_xyz": (0, 0, 0.08), "z_rotation_samples": 4},
        ],
    },
    "red_cup": {
        "prompt": "red cup", "mesh": "models/meshes/red_cup/red_cup.obj",
        "dimensions": (0.0714, 0.0728, 0.1085),
        "mjcf": {"type": "primitive", "primitive": "cup", "z_offset": -0.05425},
        "physics": {"mass": 0.25, "friction": 0.3},
        "grasps": [
            {"offset_xyz": (0, 0.03, 0.14575), "offset_rpy_deg": (180, 0, 90), "hover_offset_xyz": (0, 0, 0.05)},
            {"offset_xyz": (0, 0.03, 0.14575), "offset_rpy_deg": (180, 0, -90), "hover_offset_xyz": (0, 0, 0.05)},
            {"offset_xyz": (0, -0.03, 0.14575), "offset_rpy_deg": (180, 0, 90), "hover_offset_xyz": (0, 0, 0.05)},
            {"offset_xyz": (0, -0.03, 0.14575), "offset_rpy_deg": (180, 0, -90), "hover_offset_xyz": (0, 0, 0.05)},
        ],
    },
    "green_cup": {
        "prompt": "green cup", "mesh": "models/meshes/green_cup/green_cup.obj",
        "dimensions": (0.0729, 0.0736, 0.1115),
        "mjcf": {"type": "primitive", "primitive": "small_cup", "z_offset": -0.05575},
        "physics": {"mass": 0.25, "friction": 0.3},
        "grasps": [
            {"offset_xyz": (0, 0.03, 0.14575), "offset_rpy_deg": (180, 0, 90), "hover_offset_xyz": (0, 0, 0.05)},
            {"offset_xyz": (0, 0.03, 0.14575), "offset_rpy_deg": (180, 0, -90), "hover_offset_xyz": (0, 0, 0.05)},
            {"offset_xyz": (0, -0.03, 0.14575), "offset_rpy_deg": (180, 0, 90), "hover_offset_xyz": (0, 0, 0.05)},
            {"offset_xyz": (0, -0.03, 0.14575), "offset_rpy_deg": (180, 0, -90), "hover_offset_xyz": (0, 0, 0.05)},
        ],
    },
    "blue_cup": {
        "prompt": "blue cup", "mesh": "models/meshes/blue_cup/blue_cup.obj",
        "dimensions": (0.0728, 0.0723, 0.1104),
        "mjcf": {"type": "primitive", "primitive": "large_cup", "z_offset": -0.0552},
        "physics": {"mass": 0.25, "friction": 0.3},
        "grasps": [
            {"offset_xyz": (0, 0.03, 0.14575), "offset_rpy_deg": (180, 0, 90), "hover_offset_xyz": (0, 0, 0.05)},
            {"offset_xyz": (0, 0.03, 0.14575), "offset_rpy_deg": (180, 0, -90), "hover_offset_xyz": (0, 0, 0.05)},
            {"offset_xyz": (0, -0.03, 0.14575), "offset_rpy_deg": (180, 0, 90), "hover_offset_xyz": (0, 0, 0.05)},
            {"offset_xyz": (0, -0.03, 0.14575), "offset_rpy_deg": (180, 0, -90), "hover_offset_xyz": (0, 0, 0.05)},
        ],
    },
}


BENCHMARK_CONFIGS = {
    'mug_on_plate': {
        'active': ('red_mug_big',),
        'passive': ('blue_plate_big',),
        'instruction': ('Place the red mug on the very center of the blue plate.',),
        'default_validation_program': {
            'validation_mode': 'settling', 'initial_max_pen': 0.01, 'final_max_pen': 0.001,
            'required_contacts': ['active_passive'], 'forbidden_contacts': ['active_table'],
            'xyz_relations': [{'axis': 'z', 'relation': 'greater_than', 'subject': 'active', 'reference': 'passive'}],
            'allowed_position_drift_fraction': 0.5, 'allowed_orientation_drift_deg': 20,
        },
        'render_config': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (-1, -1, 2), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (1, 1, 0.5), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
        ),
    },
    'mug_on_rack': {
        'active': ('red_mug_big',),
        'passive': ('wooden_stick_toy',),
        'instruction': ('Hang the red mug on the top branch of the wooden rack by its handle. The handle should be facing the camera.',),
        'default_validation_program': {
            'validation_mode': 'settling', 'initial_max_pen': 0.015, 'final_max_pen': 0.0001,
            'required_contacts': ['active_passive'], 'forbidden_contacts': ['active_table'],
            'xyz_relations': [{'axis': 'z', 'relation': 'greater_than', 'subject': 'active', 'reference': 'passive'}],
            'allowed_position_drift_fraction': 0.5, 'allowed_orientation_drift_deg': 20.0,
        },
        'render_config': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (4.0, -6.0, 6.0), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (4.0, 6.0, 4.0), 'up': (0.0, 0, 1.0), 'zoom': 1.2},
        ),
        'preinsertion': ((0, -0.05, 0),),
    },
    'plate_on_stand': {
        'active': ('green_plate_small',),
        'passive': ('wooden_rack',),
        'instruction': ('Place the green plate vertically so that its rim is inserted into the same corresponding slot on both opposing rails, with the rim passing through the pair of slots',),
        'default_validation_program': {
            'validation_mode': 'settling', 'initial_max_pen': 0.2, 'final_max_pen': 0.001,
            'required_contacts': ['active_passive'], 'forbidden_contacts': ['active_table'],
            'xyz_relations': [{'axis': 'z', 'relation': 'greater_than', 'subject': 'active', 'reference': 'passive'}],
            'allowed_position_drift_fraction': 0.5, 'allowed_orientation_drift_deg': 22.0,
        },
        'render_config': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.4, 0.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 0.9},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -0.4, 0.4), 'up': (0.0, 0.0, 1.0), 'zoom': 0.8},
        ),
    },
    'mug_on_stand': {
        'active': ('red_mug_big',),
        'passive': ('wooden_rack',),
        'instruction': ("Place the red mug on the wooden stand upside down on the stand's slots. The mug should cover slots on one rail, not both rails.",),
        'default_validation_program': {
            'validation_mode': 'settling', 'initial_max_pen': 0.2, 'final_max_pen': 0.001,
            'required_contacts': ['active_passive'], 'forbidden_contacts': ['active_table'],
            'xyz_relations': [{'axis': 'z', 'relation': 'greater_than', 'subject': 'active', 'reference': 'passive'}],
            'allowed_position_drift_fraction': 0.5, 'allowed_orientation_drift_deg': 22.0,
        },
        'render_config': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -0.5, 0.2), 'up': (0.0, 0.0, 1.0), 'zoom': 0.9},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.5, 0.0, 0.8), 'up': (0.0, 0.0, 1.0), 'zoom': 0.9},
        ),
    },
    'pour_tea': {
        'active': ('teapot',),
        'passive': ('red_mug_big',),
        'instruction': ('Place the teapot at a pose where it can pour tea into the red mug. The spout needs to be above the mug and tilted at an angle to pour.',),
        'default_validation_program': {
            'validation_mode': 'settling', 'initial_max_pen': 0.08, 'final_max_pen': 0.001,
            'required_contacts': ['active_passive'], 'forbidden_contacts': ['active_table'],
            'xyz_relations': [{'axis': 'z', 'relation': 'greater_than', 'subject': 'active', 'reference': 'passive'}],
            'allowed_position_drift_fraction': 0.5, 'allowed_orientation_drift_deg': 20.0,
        },
        'render_config': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.5, 0.5, 0.4), 'up': (0.0, 0.0, 1.0), 'zoom': 1.0},
            {'lookat': (0.0, 0.0, 0.0), 'front': (-0.5, 0.5, 0.4), 'up': (0.0, 0.0, 1.0), 'zoom': 1.0},
        ),
    },
    'lid_on_teapot': {
        'active': ('yellow_cap',),
        'passive': ('teapot',),
        'instruction': ('Place the yellow lid on the teapot.',),
        'default_validation_program': {
            'validation_mode': 'settling', 'initial_max_pen': 0.08, 'final_max_pen': 0.001,
            'required_contacts': ['active_passive'], 'forbidden_contacts': ['active_table'],
            'xyz_relations': [{'axis': 'z', 'relation': 'greater_than', 'subject': 'active', 'reference': 'passive'}],
            'allowed_position_drift_fraction': 0.5, 'allowed_orientation_drift_deg': 20.0,
        },
        'render_config': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, 0.5, 0.8), 'up': (0.0, 0.0, 1.0), 'zoom': 1.0},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.5, 0.0, 0.8), 'up': (0.0, 0.0, 1.0), 'zoom': 1.0},
        ),
        'preinsertion':  ((0, 0, 0.05),),
    },
    # 'stack_cup': {
    #     'active': ('blue_cup',),
    #     'passive': ('red_cup',),
    #     'instruction': ('Place the blue cup inside the red cup.',),
    #     'default_validation_program': {
    #         'validation_mode': 'settling', 'initial_max_pen': 0.08, 'final_max_pen': 0.001,
    #         'required_contacts': ['active_passive'], 'forbidden_contacts': ['active_table'],
    #         'xyz_relations': [{'axis': 'z', 'relation': 'greater_than', 'subject': 'active', 'reference': 'passive'}],
    #         'allowed_position_drift_fraction': 0.5, 'allowed_orientation_drift_deg': 20.0,
    #     },
    #     'render_config': (
    #         {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -0.5, 10.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
    #         {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 0.8), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
    #     ),
    # },
    'stack_cup': {
        'active': ('green_cup',),
        'passive': ('red_cup',),
        'instruction': ('Place the green cup inside the red cup.',),
        'default_validation_program': {
            'validation_mode': 'settling', 'initial_max_pen': 0.08, 'final_max_pen': 0.001,
            'required_contacts': ['active_passive'], 'forbidden_contacts': ['active_table'],
            'xyz_relations': [{'axis': 'z', 'relation': 'greater_than', 'subject': 'active', 'reference': 'passive'}],
            'allowed_position_drift_fraction': 0.5, 'allowed_orientation_drift_deg': 20.0,
        },
        'render_config': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -0.5, 10.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 0.8), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
        'preinsertion':  ((0, 0, 0.05),),
    },

    # Multi-stage tasks
    'stack_cups': {
        'active': ('red_cup', 'green_cup'),
        'passive': ('blue_cup', 'red_cup'),
        'instruction': ('Place the red cup inside the blue cup.', 'Place the green cup inside the red cup.'),
        'default_validation_program': {
            'validation_mode': 'settling', 'initial_max_pen': 0.08, 'final_max_pen': 0.001,
            'required_contacts': ['active_passive'], 'forbidden_contacts': ['active_table'],
            'xyz_relations': [{'axis': 'z', 'relation': 'greater_than', 'subject': 'active', 'reference': 'passive'}],
            'allowed_position_drift_fraction': 0.5, 'allowed_orientation_drift_deg': 20.0,
        },
        'render_config': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -1.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.8, 0.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 1.2},
        ),
    },
    'plates_on_stand': {
        'active': ('green_plate_small', 'pink_plate_small', 'yellow_plate_small'),
        'passive': ('wooden_rack', 'wooden_rack', 'wooden_rack'),
        'instruction': ('Insert the green plate into a pair of opposing slots.', 'Insert the pink plate into a different pair of opposing slots.', 'Insert the yellow plate into a different pair of opposing slots.'),
        'default_validation_program': {
            'validation_mode': 'settling', 'initial_max_pen': 0.2, 'final_max_pen': 0.001,
            'required_contacts': ['active_passive'], 'forbidden_contacts': ['active_table'],
            'xyz_relations': [{'axis': 'z', 'relation': 'greater_than', 'subject': 'active', 'reference': 'passive'}],
            'allowed_position_drift_fraction': 0.5, 'allowed_orientation_drift_deg': 22.0,
        },
        'render_config': (
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.4, 0.0, 1.0), 'up': (0.0, 0.0, 1.0), 'zoom': 0.9},
            {'lookat': (0.0, 0.0, 0.0), 'front': (0.0, -0.4, 0.4), 'up': (0.0, 0.0, 1.0), 'zoom': 0.8},
        ),
    },

    # Non-prehensile tasks
    'plate_on_rack_nonprehensile': {},
    'book_on_stand_nonprehensile': {},
    'order_letters': {},
}

# Transformations
import numpy as np
# T_mujoco_board = np.array([ # arucoboard frame expressed in mujoco's origin frame
#     [1.0, 0.0, 0.0, -0.12],
#     [0.0, -1.0, 0.0, 0.0],
#     [0.0, 0.0, -1.0, 0.7893],
#     [0, 0, 0, 1]
# ])
T_mujoco_board = np.array([
    [-1.0, 0.0, 0.0, 0.29],
    [0.0, 1.0, 0.0, 0.525],
    [0.0, 0.0, -1.0, 0.78],
    [0, 0, 0, 1]
])
