#!/usr/bin/env python3
"""
Setup script pour audio-vad
Installation: pip install .
"""

from setuptools import setup, find_packages
import os

# Lire le README si disponible
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Audio VAD CLI Tool - Détection de segments de parole dans les fichiers audio"

setup(
    name="audio-vad-cli",
    version="1.0.0",
    description="CLI tool for Voice Activity Detection in audio files",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Audio VAD Team",
    author_email="contact@example.com",
    url="https://github.com/username/audio-vad-cli",
    
    # Packages Python
    py_modules=['audio_vad_cli'],
    
    # Script de ligne de commande
    entry_points={
        'console_scripts': [
            'audio-vad=audio_vad_cli:main',
            'vad=audio_vad_cli:main',  # Alias court
        ],
    },
    
    # Dépendances
    install_requires=[
        'torch>=1.9.0',
        'torchaudio>=0.9.0',
        'numpy>=1.19.0',
    ],
    
    # Métadonnées
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    
    python_requires='>=3.7',
    
    # Fichiers additionnels
    include_package_data=True,
    package_data={
        '': ['*.md', '*.txt', '*.yml', '*.yaml'],
    },
    
    # Mots-clés
    keywords='audio vad voice activity detection speech silence cli tool',
    
    # Projet URLs
    project_urls={
        'Bug Reports': 'https://github.com/username/audio-vad-cli/issues',
        'Source': 'https://github.com/username/audio-vad-cli',
        'Documentation': 'https://github.com/username/audio-vad-cli#readme',
    },
)