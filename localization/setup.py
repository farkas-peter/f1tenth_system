import os
from glob import glob
from setuptools import setup

package_name = 'localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='f1tenth',
    maintainer_email='peterfarkas@edu.bme.hu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'coord_trans_node = localization.coord_trans_node:main',
            'localization_vis_node = localization.localization_vis_node:main',
            'gps_eval_node = localization.gps_eval_node:main',
            'osm_goal_publisher = localization.osm_goal_publisher:main',
        ],
    },
)
