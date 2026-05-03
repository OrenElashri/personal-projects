# Practical_Work

Small Object Detection & Dataset Analysis Benchmark
A comprehensive benchmarking toolkit for evaluating modern object detection models with a specific focus on small object detection accuracy and inference speed across COCO and OpenImages.

The project combines dataset acquisition, small-object distribution analysis, FP16 benchmarking, and automatic visualization into one workflow. It supports side-by-side comparison of YOLOv5u, YOLO11, RT-DETR, and RT-DETRv2 model families, and stores outputs in structured experiment folders for reproducible analysis. 
​

Overview
Small objects remain one of the hardest regimes in object detection, especially when comparing datasets with different class distributions and annotation characteristics. This project is designed to measure that gap directly by analyzing object-size distributions and benchmarking model performance with metrics such as AP, AP small, AP50 small, latency, and throughput under FP16 or mixed-precision inference. 
​

In the attached benchmark summary, the toolkit reports COCO small-object AP values up to 0.3527 for YOLO11x and OpenImages small-object AP values up to 0.0745 for RT-DETR-X, illustrating both the challenge of the task and the importance of cross-dataset evaluation. 
​

Key Features
Unified downloader for preparing datasets and pretrained weights from a single entry point.

Dataset analysis pipeline for measuring the frequency of small objects, defined here as objects with area smaller than $32^2$ pixels.

Performance benchmarking across multiple detector families using AP, AP small, AP50 small, latency, and speed-focused evaluation. 
​

Automated chart generation for normalized class ratios, log-scale count plots, and AP-small versus latency visualizations.

Structured experiment outputs with dated runtime folders for easier comparison and traceability. 
​

Project Workflow
1. Download datasets and model weights
Use the unified downloader to prepare the environment and fetch the required assets.

bash
python download_datasets_pw.py
Expected dependencies include:

bash
pip install fiftyone ultralytics transformers torch
The downloader is intended to simplify setup for MS COCO 2017, OpenImages V7, and pretrained detector checkpoints used in the benchmark suite.

2. Analyze dataset composition
Run the dataset analysis script to inspect how often small objects appear overall and per class.

bash
python dataset_analysis_pw.py
This stage generates statistics and charts in experiment folders named with the execution date and runtime ID, making repeated analyses easier to track over time.

3. Run the benchmark
Execute the benchmark script to evaluate model quality and speed on small-object detection.

bash
python small_object_benchmark_pw.py
The benchmark emphasizes AP small and AP50 small, while also reporting broader detection quality and runtime measurements such as AP all and latency. 
​

Supported Models
Architecture	Source	Versions
YOLOv5u	Ultralytics	n, s, m, l, x
YOLO11	Ultralytics	n, s, m, l, x
RT-DETR	Ultralytics / Hugging Face	l, x / r50, r101
RT-DETRv2	Hugging Face	r18, r50, r101
These families cover lightweight to larger-capacity models, making the project suitable for both edge-speed experiments and higher-accuracy comparisons.

Benchmark Metrics
The main evaluation target is small-object detection, with a small object defined as an instance whose area is below 32 x 32 pixels. The benchmark records the following core metrics: 
​

AP (all): overall average precision across evaluated detections.

AP (small): average precision restricted to small-object instances. 
​

AP50 (small): small-object precision at IoU 0.50.

Latency: per-image inference time in milliseconds. 
​

FPS / throughput: speed-oriented measurement for practical deployment analysis.

Example Findings
Results from the attached summary report show a clear trade-off between speed and small-object accuracy, as well as a substantial dataset gap between COCO and OpenImages. On COCO, YOLO11x reached AP small 0.3527 at 4.41 ms/image, while RT-DETR-L reached 0.3216 at 7.13 ms/image. On OpenImages, RT-DETR-X achieved the highest reported AP small at 0.0745 with 10.29 ms/image latency. 
​

The same report also highlights severe cross-dataset performance drops for small-object AP, with relative drops commonly around 70 to 82 percent when moving from COCO to OpenImages for matched classes. That makes dataset analysis an essential part of interpreting benchmark results rather than a separate preprocessing step. 
​

Output Organization
All benchmark artifacts are stored in structured output folders such as small_object_benchmark_fp16_DATE/. These folders can include prediction JSON files, evaluation metrics, summary reports, and generated figures for direct visual comparison across runs. 
​

A typical analysis workflow may therefore produce:

Raw prediction outputs.

Aggregated metric summaries.

Class-level small-object distribution charts.

AP-small versus latency plots.

Runtime-stamped experiment directories for reproducibility.

Repository Layout
A practical top-level layout for the repository can look like this:

text
.
├── download_datasets_pw.py
├── dataset_analysis_pw.py
├── small_object_benchmark_pw.py
├── experiments/
│   ├── dataset_analysis_<DATE>_<RUN_ID>/
│   └── small_object_benchmark_fp16_<DATE>_<RUN_ID>/
├── outputs/
├── weights/
└── README.md
Use Cases
This project is useful for:

Comparing detector families under small-object-heavy evaluation settings.

Studying annotation and size-distribution differences between COCO and OpenImages.

Selecting models for latency-constrained deployments.

Building reproducible internal benchmarks for research or production evaluation.

Notes
The attached visualizations suggest that class composition and small-object prevalence differ strongly across datasets, which helps explain why the same model family can rank differently on COCO and OpenImages. For rigorous comparison, keep preprocessing, precision mode, hardware, and matched-class definitions fixed across runs. 
​

