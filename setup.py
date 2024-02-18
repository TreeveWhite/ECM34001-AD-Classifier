from setuptools import setup, find_packages

setup(
    name="ad_classifier",
    version="1.2.0",
    author="Treeve White",
    author_email="treevew@gmail.com",
    description="AD Model",
    packages=find_packages('src'),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[

    ],

    zip_safe=False
)
