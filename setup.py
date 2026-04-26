from setuptools import setup, find_packages

setup(
    name='caries-detr',
    version='1.0.0',
    description=(
        'Tooth Structure-aware Prior and Lesion-aware Dynamic Loss Refinement '
        'for DETR Based Caries Detection'
    ),
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Xuefen Liu',
    url='https://github.com/XuefenLiu-SZU/caries-detr',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.10.0',
        'torchvision>=0.11.0',
        'mmengine>=0.7.0',
        'mmcv>=2.0.0',
        'mmdet>=3.0.0',
        'numpy>=1.20.0',
        'pycocotools>=2.0.4',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
)
