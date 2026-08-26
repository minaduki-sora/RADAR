from setuptools import find_packages, setup


INSTALL_REQUIRES = [
    "torch==2.6.0",
    "transformers>=4.47.0",
    "accelerate==0.26.0",
    "fschat==0.2.31",
    "gradio==3.50.2",
    "openai==0.28.0",
    "anthropic==0.5.0",
    "sentencepiece==0.1.99",
    "protobuf==3.19.0",
    "datasets>=2.14.0",
    "huggingface-hub>=0.24.0",
    "matplotlib>=3.7.0",
    "numpy>=1.24.0",
    "safetensors>=0.4.0",
    "seaborn>=0.12.0",
    "shortuuid>=1.0.0",
    "tqdm>=4.65.0",
    "wandb>=0.15.0",
]

setup(
    name='radar',
    version='1.0.0',
    description='Accelerating LLMs by 3x with No Quality Loss',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author_email='mahiru@mail.ustc.edu.cn',
    url='https://github.com/minaduki-sora/RADAR',
    license='Apache-2.0',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
)
