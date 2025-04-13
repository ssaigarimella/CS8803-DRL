from setuptools import setup, find_packages

setup(
    name='pilco',
    version='0.1',
    packages=find_packages(include=['pilco', 'pilco.*']),
    install_requires=[
        'numpy',
        'torch',
        'gpytorch',
        'matplotlib',
        'scipy',
        'gymnasium[mujoco]>=0.29.1',  # Includes Gymnasium + MuJoCo bindings
    ],
    author='nrontsis',
    description='PILCO with GPyTorch backend for probabilistic model-based reinforcement learning',
    python_requires='>=3.7',
)
