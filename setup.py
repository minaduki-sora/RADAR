from setuptools import setup, find_packages

setup(
    name='radar',
    version='1.0.0',
    description='Accelerating LLMs by 3x with No Quality Loss',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='mahiru@mail.ustc.edu.cn',
    url='https://github.com/minaduki-sora/RADAR',
    packages=find_packages(),
    install_requires=[
        'torch==2.0.1',
        'transformers==4.46.2',
        'accelerate==0.21.0',
        'fschat==0.2.31',
        'gradio==3.50.2',
        'openai==0.28.0',
        'anthropic==0.5.0',
        'sentencepiece==0.1.99',
        'protobuf==3.19.0',
        'wandb'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
)
