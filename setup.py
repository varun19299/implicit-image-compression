import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="implicit_image", # Replace with your own username
    version="0.1",
    author="Varun Sundar, Megh Doshi, Zach Huemann",
    author_email="vsundar4@wisc.edu",
    description="Implicit Image compression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/varun19299/implicit-image-compression.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)