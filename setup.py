from setuptools import setup, find_packages

setup(
    name="ad_classifier",
    version="0.0.1",
    author="Treeve White",
    author_email="treevew@gmail.com",
    description="AD Model",
    packages=find_packages('src'),
    package_data={"": "src"},
    include_package_data=True,
    install_requires=[

    ],

    zip_safe=False
)
