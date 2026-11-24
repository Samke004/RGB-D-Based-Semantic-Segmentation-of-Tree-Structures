# RGB-D Based Semantic Segmentation of Tree Structures for Precision Agriculture and Robotic Manipulation

This repository is the official implementation for the paper **"RGB-D Based Semantic Segmentation of Tree Structures for Precision Agriculture and Robotic Manipulation"** by **Samuel Adžić**, **Jana Dukić**, and **Emmanuel Karlo Nyarko** (*Faculty of Electrical Engineering, Computer Science and Information Technology Osijek, Croatia*).

## Abstract
Robotic pruning of dormant orchards requires robust semantic segmentation, yet RGB-D methods are predominantly benchmarked on dense indoor scenes. We evaluate 15 configurations across three model families on 663 annotated RGB-D images of trellis-grown pear trees (distinguishing **trunk**, **branches**, **supporting structure**, and **grass**) evaluated under a standardized single-scale protocol across two tree-disjoint data partitions. Our findings show that U-Net (ResNet-34) and ESANet achieve top performance (91.8–92.0 mIoU), but remain statistically indistinguishable given random seed variance (up to 1.09 points). Due to sparse field geometry, RGB-D pretraining does not provide measurable advantage over conventional ImageNet initialization. Two rarely stated reporting choices—whether background enters the mean and whether inference is multi-scale—move the numbers by 1.4 to 2.7 points, several times the differences they would be used to compare. For practical deployment, high-accuracy perception does not require heavy computational overhead; lightweight models like DeepLabV3+ (MobileNetV2) achieve competitive accuracy at a fraction of the latency. These results demonstrate that orchard pruning robots can achieve real-time, high-accuracy perception using compact, RGB-pretrained encoders.

## Key Highlights
- **Architectures Evaluated:** U-Net, ESANet, and DeepLabV3+ across 15 configurations.
- **Dataset:** 663 annotated RGB-D images of dormant trellis-grown pear trees evaluated across two tree-disjoint splits.
- **Classes:** Trunk, branches, supporting structure, and grass.
- **Practical Takeaway:** Compact, ImageNet-pretrained models (e.g., DeepLabV3+ with MobileNetV2) deliver real-time, high-accuracy perception suitable for orchard pruning robotics without heavy RGB-D pretraining overhead.
