from setuptools import setup

package_name = 'coord_trans'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'coord_trans_node = coord_trans.coord_trans_node:main',
            'localization_vis_node = coord_trans.localization_vis_node:main',
            'gps_eval_node = coord_trans.gps_eval_node:main'
        ],
    },
)
