from setuptools import setup, find_packages

setup(
    name="sistema_recomendacao_api",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pandas"
    ],
)