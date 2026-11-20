from glob import glob

from setuptools import find_packages, setup

package_name = 'planisferio'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Raymundo Gomez',
    maintainer_email='raymundo.gomez.rv@gmail.com',
    description='2D LiDAR SLAM for ROS 2: particle-filter front-end, pose-graph back-end '
                'with loop closure.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'slam = planisferio.slam_node:main',
            'benchmark = planisferio.benchmark:main',
        ],
    },
)
