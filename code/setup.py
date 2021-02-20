from setuptools import setup, find_packages

# import versioneer

setup(
    name="romtime",
    # version=versioneer.get_version(),
    # cmdclass=versioneer.get_cmdclass(),
    description="Reduced Order Model generator for FEM problems.",
    author_email="enrique.millanvalbuena@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
)