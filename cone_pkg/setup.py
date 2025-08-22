from setuptools import setup

package_name = 'cone_pkg'

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
    maintainer_email='csaba.boros@edu.bme.hu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drive_publisher = cone_pkg.drive_publisher:main',
            'pure_pursuit_control = cone_pkg.pure_pursuit_control:main',
            'intel_yolo = cone_pkg.intel_yolo:main',
            'data_logger = cone_pkg.data_logger:main',
            'ekf = cone_pkg.ekf:main',
            'FSS = cone_pkg.FSS:main'
        ],
    },
)
