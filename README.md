# Multi-Resampled Image Retrieval Tool

A C++ tool for **retrieving images that have been resampled multiple times**, designed for dataset auditing, preprocessing validation, and computer vision pipeline debugging.

## Overview

This tool is specifically built to **identify and retrieve images that have undergone multiple resampling operations**, such as repeated resizing, scaling, or resolution changes.

Unlike typical image processing utilities, this project:
- **Does NOT perform resampling**
- Focuses purely on **detecting and retrieving images that were resampled multiple times**

It is intended for engineering and algorithm workflows where excessive resampling may introduce artifacts, degrade image quality, or affect model performance.

## Key Capabilities

- Detect images that have been **resampled more than once**
- Retrieve and export multi-resampled images from large datasets
- Designed for **high-performance C++ pipelines**
- Suitable for offline batch analysis and automated data checks

## Typical Use Cases

- Dataset quality control and auditing
- Detecting excessive resize operations in data pipelines
- Identifying image degradation caused by repeated preprocessing
- Debugging data augmentation or conversion workflows
- Regression testing for image preprocessing modules

## Concept

In this project:

- **Resampled image**  
  An image that has been resized, scaled, or spatially transformed.

- **Multi-resampled image**  
  An image that has undergone **multiple resampling steps**, potentially accumulating interpolation artifacts.

- **Retrieval**  
  Locating and extracting such images from a larger dataset for inspection or downstream processing.

## Workflow

1. Load images from a directory or dataset
2. Analyze image characteristics (resolution changes, metadata, pixel-level patterns, etc.)
3. Identify candidates that indicate **multiple resampling**
4. Retrieve or export matched images

## Build

### Command Line

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
cd ..
```

## Usage

### Command Line

```bash
./build/Release/demo.exe
```
