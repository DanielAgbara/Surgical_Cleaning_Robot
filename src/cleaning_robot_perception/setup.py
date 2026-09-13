from setuptools import find_packages, setup
from glob import glob
import os


package_name = 'cleaning_robot_perception'


setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(
        exclude=['test']
    ),

    data_files=[
        # Register package with ament
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),

        # Install package.xml
        (
            'share/' + package_name,
            ['package.xml']
        ),

        # Install launch files
        (
            os.path.join(
                'share',
                package_name,
                'launch'
            ),
            glob('launch/*.launch.py')
        ),

        # Install configuration files
        #
        # YAML:
        #   future ROS parameter files
        #
        # JSON:
        #   marker_to_cleaning_head.json
        (
            os.path.join(
                'share',
                package_name,
                'config'
            ),
            (
                glob('config/*.yaml')
                + glob('config/*.json')
            )
        ),

        # Install Detectron2 tray model
        (
            os.path.join(
                'share',
                package_name,
                'models',
                'maskrcnn_tray'
            ),
            glob('models/maskrcnn_tray/*')
        ),
    ],

    install_requires=[
        'setuptools'
    ],

    zip_safe=True,

    maintainer='agbara-admin',
    maintainer_email='daniel.agbara17@gmail.com',

    description=(
        'Perception package for the '
        'Surgical Cleaning Robot'
    ),

    license='Apache-2.0',

    extras_require={
        'test': [
            'pytest',
        ],
    },

    # --------------------------------------------------------
    # Python executable
    # --------------------------------------------------------
    #
    # Use the python3 currently active on PATH.
    #
    # When the cleaning_robot Conda environment is active:
    #
    #   /usr/bin/env python3
    #
    # resolves to:
    #
    #   ~/miniconda3/envs/cleaning_robot/bin/python3
    #
    # This allows ROS nodes to access:
    #
    #   PyTorch
    #   Detectron2
    #   PyZED
    #   OpenCV
    #   Open3D
    #   NumPy
    #
    options={
        'build_scripts': {
            'executable': '/usr/bin/env python3',
        },
    },

    # --------------------------------------------------------
    # ROS executables
    # --------------------------------------------------------

    entry_points={
        'console_scripts': [
            (
                'zed_camera = '
                'cleaning_robot_perception.zed_camera:main'
            ),
            (
                'object_detector = '
                'cleaning_robot_perception.object_detector:main'
            ),
            (
                'plane_detector = '
                'cleaning_robot_perception.plane_detector:main'
            ),
            (
                'charuco_tracker = '
                'cleaning_robot_perception.charuco_tracker:main'
            ),
            (
                'plot_marker_transform = '
                'cleaning_robot_perception.'
                'plot_marker_transform:main'
            ),
            (
                'edit_marker_transform = '
                'cleaning_robot_perception.'
                'edit_marker_transform:main'
            ),
        ],
    },
)