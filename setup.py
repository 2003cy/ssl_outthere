from setuptools import setup, find_packages
from pathlib import Path

README = Path(__file__).with_name("README.md")
long_description = README.read_text(encoding="utf-8") if README.exists() else ""
REQ_FILE = Path(__file__).with_name("requirements.txt")

def load_requirements(path: Path):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    reqs = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        reqs.append(ln)
    return reqs

setup(
    name="ssl_outthere",
    version="0.0.1",
    description="Multi modal, SSL representation learning and downstream tasks for JWST NIRISS/NIRCam images and spectra",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yang Cheng",
    author_email="yacheng@mpia.de",
    license="MIT",
    packages=find_packages(exclude=("images", "outputs", "notebooks", "data")),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=load_requirements(REQ_FILE),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)