from setuptools import find_packages, setup
from glob import glob

package_name = "playground"

map_files = [
    f for pattern in ("map*/**/*.pgm", "map*/**/*.yaml")
    for f in glob(pattern, recursive=True)
]

data_files = [
    ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
    (f"share/{package_name}", ["package.xml"]),
    (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    (f"share/{package_name}/worlds", glob("worlds/*")),
    (f"share/{package_name}/models", ["models/policy.pth"]),
    (f"share/{package_name}/map", map_files),
]

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sim_environment = playground.sim_environment:main",
            "real_environment = playground.real_environment:main",
        ],
    },
)
