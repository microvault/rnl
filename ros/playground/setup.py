from setuptools import find_packages, setup

package_name = "playground"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/turtlebot_world.launch.py"]),
        ("share/" + package_name + "/worlds", ["worlds/my_world.world"]),
        ("share/" + package_name + "/models", ["models/model.zip"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["environment = playground.environment:main"],
    },
)
