# Semantic Segmentation

## 1. What is Semantic Segmentation?

**Semantic segmentation** is the pixel-level classification task: given an input image, assign each pixel a class label (e.g., road, person, sky, car). Unlike image classification (one label per image) and object detection (bounding boxes), semantic segmentation produces a dense class map of the same spatial dimensions as the input.

Key property: all pixels of the same semantic class are treated equally (no instance separation). For example, two different cars are both labeled “car” without distinguishing which pixel belongs to which instance.

Related tasks:

- **Instance segmentation**: distinguishes individual object instances (e.g., car-1, car-2).
- **Panoptic segmentation**: unifies semantic and instance segmentation — full scene understanding (semantic labels for stuff + instance labels for things).

---

## 2. Problem formulation (mathematics)

Given image (X \in \mathbb{R}^{H\times W\times C}) (height H, width W, channels C), semantic segmentation produces prediction (\hat{Y} \in {1..K}^{H\times W}) (K classes). Training data: images and per-pixel ground-truth (Y).

Typical network outputs logits (z*{i,c}) per pixel i and class c; softmax gives class probabilities:
[
p*{i,c} = \frac{\exp(z*{i,c})}{\sum*{c'} \exp(z*{i,c'})}
]
Loss typically is pixel-wise cross-entropy:
[
L = -\frac{1}{N}\sum*{i}\sum*{c} \mathbf{1}[y_i=c] \log p*{i,c}
]
where (N = H\times W) (or batch sum).

Because of class imbalance and region-based importance, other losses (IoU/Dice/Lovász/focal) are used or combined — see losses section.

---

## 3. Classic architectures & design patterns

### 3.1 Fully Convolutional Network (FCN)

- First idea to convert classification nets to dense predictors by replacing FC layers with convolutions and upsampling (long-skip connections from earlier layers to recover spatial detail). Produces coarse logits then upsample with deconvolution/bilinear.

### 3.2 Encoder–Decoder style (U-Net, SegNet)

- **Encoder**: stack of conv + pool (backbone CNN) extracts features at lower spatial resolution.
- **Decoder**: series of upsampling steps (transpose conv / unpool / interpolation) to recover original resolution.
- **U-Net**: symmetric encoder–decoder with skip connections concatenating encoder feature maps to decoder (preserves fine-grained localization).
- **SegNet**: uses pooling indices for unpooling (less parameters than learned deconv).

### 3.3 Atrous/Dilated Convolutions & DeepLab family

- **Atrous (dilated) convolution** increases receptive field without downsampling by inserting holes (dilation) in kernels.
- **ASPP (Atrous Spatial Pyramid Pooling)**: parallel atrous convs with different dilation rates to capture multiscale context.
- **DeepLab (v1..v3+)**: combines atrous convs, ASPP, and often decoder/refinement modules.

### 3.4 Pyramid Pooling (PSPNet)

- Pools the feature map at several grid scales then upsamples and concatenates to add global context (multi-scale pooling idea).

### 3.5 Modern backbones & hybrid models

- Backbones (ResNet, EfficientNet, MobileNet) serve as encoder. Vision transformers (ViT/SegFormer) and hybrid CNN-Transformer backbones are increasingly used for better global context.

---

## 4. Upsampling methods (decoder choices)

- **Nearest / bilinear interpolation** (fast, no learnable params).
- **Transposed convolution (deconvolution)** (learnable, can introduce checkerboard artifacts if misused).
- **Unpooling** using saved pooling indices (SegNet).
- **Sub-pixel convolution** (PixelShuffle) for super-resolution-like recovery.
- **Learned upsampling + conv** often used: upsample (bilinear) → conv → non-linearity.

Design trade-offs: learned upsampling gives higher accuracy but costs parameters and computation; interpolation is memory- and compute-efficient.

---

## 5. Loss functions & optimization specifics

### 5.1 Pixel-wise Cross-Entropy

Standard baseline. Weighted versions (per-class weights) help with class imbalance.

### 5.2 Intersection over Union (IoU / Jaccard)

Often used as metric, can be converted to loss (1 - IoU). IoU for class c:
[
IoU_c = \frac{TP_c}{TP_c + FP_c + FN_c}
]
Mean IoU (mIoU) is average across classes.

### 5.3 Dice / F1 Loss

Dice coefficient:
[
Dice = \frac{2\sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}
]
Dice loss (=1 - Dice). Useful for imbalanced foreground (medical).

### 5.4 Focal Loss

Down-weights easy negatives and focuses training on hard pixels (useful when background dominates).

### 5.5 Lovász-Softmax Loss

A differentiable surrogate that directly optimizes IoU; useful for mIoU improvement.

### 5.6 Combined losses

Common to sum cross-entropy + Dice or IoU loss to combine pixel accuracy and region overlap objectives.

---

## 6. Metrics & evaluation

- **Pixel Accuracy**: proportion of correctly labeled pixels (can be dominated by large background).
- **Per-class Accuracy**: accuracy per class, then average.
- **Mean IoU (mIoU)**: the standard metric for segmentation competitions.
- **F1 / Dice score**: especially in medical segmentation.
- **Boundary metrics**: evaluate contour/boundary quality, e.g., boundary F1, Hausdorff distance (medical).
- **Panoptic Quality (PQ)**: for panoptic segmentation, combines recognition and segmentation quality.

Evaluation tips:

- Report per-class IoU and mIoU.
- Use confusion matrix to analyze which classes are confused.

---

## 7. Data, annotations, and datasets

Annotation is expensive: per-pixel labels are labor-intensive. Common datasets:

- **PASCAL VOC** (20 classes)
- **Cityscapes** (urban street scenes) — high-res pixel labels
- **ADE20K** — dense scene parsing dataset with many classes
- **COCO Stuff** / COCO Panoptic — instance + stuff labels
- **CamVid, KITTI** for driving
- **Medical datasets** (ISIC, BRATS, etc.)
  Annotation formats: PNG masks (per-pixel id), COCO JSON (RLE for instance masks), Pascal VOC XML+mask.

Labeling tools: LabelMe, COCO Annotator, CVAT. Semi-automatic tools, interactive segmentation (GrabCut), and active learning can reduce cost.

---

## 8. Practical training strategies and tips

### 8.1 Pretraining & transfer learning

Use ImageNet-pretrained backbones; they accelerate convergence and improve accuracy. Fine-tune decoder layers.

### 8.2 Crop size and batch size

Large images cause memory issues — common to train with random crops (e.g., 512×512) and batch sizes tuned to GPU memory. Use synchronized BN (SyncBN) across GPUs for small per-GPU batches.

### 8.3 Data augmentation

Critical: random scale (0.5–2.0), random crop, horizontal flip, color jitter, Gaussian noise, CutMix/MixUp variants adapted for segmentation. Multi-scale training improves robustness.

### 8.4 Optimizers & LR schedules

SGD + momentum with poly learning rate (power decay) is a standard; Adam or AdamW also used. Warmup for large batch training, and careful weight decay (often decoupled from Adam, e.g., AdamW).

### 8.5 Class imbalance handling

Use class weighting in cross-entropy, focal loss, oversampling rare classes, or boundary-aware losses.

### 8.6 Post-processing

- **CRF (DenseCRF)** or bilateral filtering refines boundaries (used historically).
- For instance-aware tasks, NMS / association heuristics.
- Test-time augmentation (multi-scale + flips) and ensembling improve results.

---

## 9. Challenges & failure modes

- **Boundary precision**: recovering crisp edges is hard — skip connections, decoder design, and CRF help.
- **Small objects**: downsampling hurts small-object detection — use multi-scale features, dilated convs, FPN-style pyramids.
- **Class imbalance**: background dominates; design losses and sampling accordingly.
- **Memory & speed**: high-res segmentation is expensive — need efficient backbones (MobileNet, EfficientNet), depthwise separable convs, or pruning/quantization for deployment.
- **Domain shift**: models trained in one domain (simulator) may fail in real-world — domain adaptation, unsupervised / self-supervised pretraining help.
- **Annotation noise**: inconsistent masks degrade training; label smoothing or robust losses mitigate.

---

## 10. Advanced topics & research directions

### 10.1 Multi-scale & feature pyramids

Feature Pyramid Network (FPN) combines multiple scales to detect both small and large structures.

### 10.2 Attention & transformers

Self-attention captures global context; recent approaches (SegFormer, SETR) use transformers or hybrid CNN-transformer backbones for long-range dependencies.

### 10.3 Weakly-supervised / semi-supervised segmentation

Using image-level labels, bounding boxes, or scribbles plus consistency regularization, pseudo-labeling, or CRF-based propagation to reduce labeling cost.

### 10.4 Domain adaptation & self-supervision

Adversarial training or self-supervised pretraining for robust cross-domain performance (synthetic → real).

### 10.5 Real-time segmentation

Models like ENet, BiSeNet, Fast-SCNN aim for fast inference (autonomous driving, robotics).

### 10.6 Panoptic segmentation

Unified segmentation that jointly handles stuff (semantic) and things (instances) — requires combined architectures and metrics (PQ).

---

## 11. Implementation checklist (practical)

- Choose a strong backbone (ResNet/ResNeSt/EfficientNet) pretrained on ImageNet.
- Use encoder–decoder with skip connections (U-Net / FPN) or DeepLab/PSP-style modules for context.
- Train with adequate augmentation (scales, flips, color jitter).
- Use combined loss (cross-entropy + Dice/Lovász) for robust optimization.
- Schedule learning rate (poly, cosine annealing, one-cycle).
- Monitor mIoU and per-class IoU; save best model by validation mIoU.
- Use multi-scale inference and CRF or learned refinement if boundary precision is critical.
- For deployment, optimize using quantization/pruning or lightweight backbones.

---

## 12. Example: Why U-Net works well for medical segmentation

- U-Net’s symmetric encoder-decoder with skip connections preserves high-resolution localization and simultaneously provides deep semantic context — ideal when small structures matter (tumors, vessels). Dice loss commonly used because foreground is small and class-imbalanced.

---

## 13. Summary

Semantic segmentation transforms images into dense, per-pixel labels and sits at the core of scene understanding. The task is both conceptually simple and technically deep: you must balance local detail (boundaries, small objects) with global context (object-level semantics). Key ingredients are appropriate architectures (encoder-decoder, atrous convs, ASPP/PSP), suitable losses (cross-entropy, Dice, IoU), strong data augmentation, careful training (LR schedules, batchnorm, pretrained backbones), and post-processing for boundary refinement. Modern research extends classical CNN paradigms with attention, transformers, weak supervision, and efficiency methods for real-time deployment.

---

---

---

# Object Detection

**Object detection** is the computer vision task of **locating** and **classifying** objects in images (or video). For each object instance the model outputs a **bounding box** (usually an axis-aligned rectangle) and a **class label** (and often a confidence score). Object detection sits between image classification (one label per image) and semantic/instance segmentation (pixel-level masks) and is fundamental for applications like autonomous driving, surveillance, robotics, retail analytics, and AR.

Below is a thorough, nitty-gritty guide covering definitions, formulations, architectures, losses, training recipes, evaluation, practical considerations, and advanced topics.

---

## 1. Problem formulation & outputs

Given an image (I) of size (H \times W \times C), the detector must produce a set of detections:
[
{(c_k, s_k, x_k, y_k, w_k, h_k)}_{k=1}^M
]
where:

- (c_k): predicted class (or class probabilities),
- (s_k): confidence score (objectness or class confidence),
- ((x_k, y_k)): box center coordinates,
- ((w_k, h_k)): box width and height (often normalized to image size).

Boxes can be encoded in different ways (corner coordinates ((x*\text{min}, y*\text{min}, x*\text{max}, y*\text{max})), or center + size). During training we match predicted boxes to ground-truth boxes and optimize a multi-task loss (classification + localization).

---

## 2. Key concepts and metrics

### Intersection over Union (IoU)

[
IoU(B_p, B_{gt}) = \frac{\text{area}(B_p \cap B_{gt})}{\text{area}(B_p \cup B_{gt})}
]
Used to evaluate overlap between predicted box (B*p) and ground-truth (B*{gt}). IoU thresholds (e.g., 0.5) decide if a prediction is a true positive.

### Average Precision (AP) and mean AP (mAP)

- For each class, compute precision–recall curve by varying score threshold.
- **AP** = area under precision–recall curve (various interpolation rules exist).
- **mAP** = average of APs across classes.
- COCO uses AP averaged over multiple IoU thresholds from 0.50 to 0.95 (step 0.05), commonly denoted AP@[.50:.95], and also reports AP@0.50, AP@0.75, and size-specific metrics (AP(\_S), AP(\_M), AP(\_L)).

### Other metrics

- **Recall** at limited proposals (R@k), Average Recall (AR).
- **Precision** and **F1**.
- In deployment: latency (ms), throughput (FPS), model size, FLOPs.

---

## 3. Two major families of detectors

### A. Two-stage detectors (region-proposal based)

**Examples:** R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN

**Pipeline:**

1. **Backbone** extracts features (e.g., ResNet).
2. **Region Proposal Network (RPN)** proposes candidate object regions (class-agnostic).
3. **RoI pooling / RoI Align** extracts fixed-size features for each proposal.
4. **Head** classifies each proposal and regresses refined box coordinates (and optionally mask).

**Pros:** high accuracy, refined localization.
**Cons:** slower, more complex; historically not real-time.

### B. One-stage detectors (dense prediction)

**Examples:** YOLO family, SSD, RetinaNet, CornerNet, FCOS

**Pipeline:**

- Directly predict class scores and box coordinates on a dense grid/anchors or anchor-free locations from feature maps.
- Single network outputs detections in one pass.

**Pros:** faster, simpler pipeline suitable for real-time.
**Cons:** historically lower accuracy for small objects and class imbalance (but modern designs like RetinaNet, YOLOv4/v5/v7, EfficientDet close the gap).

---

## 4. Anchor-based vs Anchor-free detectors

### Anchor-based detectors

- Use pre-defined boxes (anchors/prior boxes/priors) at multiple scales/aspect ratios per spatial location.
- For each anchor, predict box offsets and class probabilities.
- Anchor design is crucial: scales, aspect ratios, and placements control coverage.
- **SSD**, **RetinaNet**, **Faster R-CNN** are anchor-based.

**Box encoding** typically uses transforms:
[
t_x = (x_{gt} - x_a) / w_a,\quad t_w = \log(w_{gt}/w_a)
]
and similar for y,h — then regress these targets.

### Anchor-free detectors

- Predict objects without pre-defined anchors. Approaches include:

  - **Keypoint-based:** predict object center and size (CenterNet).
  - **Dense regression:** predict box coordinates at each pixel/feature point (FCOS).
  - **Corner-based:** predict box corners as keypoints (CornerNet).

- Advantages: simpler, fewer hyperparameters, often faster. They solve some issues with anchor matching and imbalance.

---

## 5. Architecture building blocks

### Backbone (feature extractor)

- Typical: ResNet, ResNeXt, EfficientNet, MobileNet for lightweight models.
- Pretrained on ImageNet for faster convergence and better features.

### Neck

- Intermediate modules that fuse multi-scale features:

  - **FPN (Feature Pyramid Network)**: top-down pathway + lateral connections to combine low-level high-resolution with high-level semantic features. Crucial for multi-scale object detection.
  - PANet, BiFPN (used in EfficientDet) offer enhanced information flow.

### Head

- Predicts classes and boxes from fused features.
- Heads are task-specific (classification head, bbox regression head, centerness head, etc.).
- Design choices: shared vs separate heads for class & reg.

---

## 6. Loss functions & training targets

Object detection uses **multi-task loss**:
[
L = L_{cls} + \lambda L_{loc} + L_{obj/centerness} + \ldots
]

### Classification loss

- Cross-entropy for softmax multi-class.
- Typically uses **focal loss** (RetinaNet) to address class imbalance:
  [
  FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
  ]
  where (p_t) is predicted prob for target class, (\gamma) focuses learning on hard examples.

### Localization loss

- **Smooth L1 (Huber) loss** is common:
  [
  \text{smooth}_{L1}(x) = \begin{cases}
  0.5x^2 & |x|<1 \
  |x|-0.5 & \text{otherwise}
  \end{cases}
  ]
- Some works explore IoU-based losses (IoU loss), GIoU, DIoU, CIoU — these directly optimize a box overlap metric and improve localization:

  - **GIoU** adds a penalty for non-overlap using smallest enclosing box.
  - **CIoU** considers aspect ratio and center distance.

### Objectness / centerness

- For anchor-based RPNs, an objectness score indicates whether an anchor contains an object.
- Anchor-free methods like FCOS predict a **centerness** score to down-weight low-quality predictions far from object centers.

### Matching strategy & positive/negative sampling

- Anchor matches to GT if IoU > pos_threshold (e.g., 0.5), negative if < neg_threshold.
- Hard negative mining or **OHEM** selects hard negatives to balance training.
- Focal loss and adaptive sampling reduce the need for complex mining.

---

## 7. Non-Maximum Suppression (NMS) and variants

Multiple overlapping predictions must be suppressed. Standard NMS:

1. Sort detections by score.
2. Keep highest score detection, remove all with IoU > threshold.
3. Repeat.

Problems:

- NMS is class-agnostic unless applied per class.
- Hard threshold can remove true objects in crowded scenes.

Variants:

- **Soft-NMS**: reduces scores of overlapping boxes rather than removing them; better for crowded scenes.
- **Class-wise NMS**: perform per-class suppression.
- Batched and GPU-optimized NMS implementations are common for speed.

---

## 8. Multi-scale detection & handling small objects

Small objects are challenging: downsampling in backbones and pooling can erase information. Strategies:

- **Use FPN** so high-resolution features contribute.
- **Detect at multiple scales** (SSD, YOLO use different feature map levels).
- **Reduce stride** in early layers or use dilated convolutions to increase resolution.
- **Super-resolution / magnification** or multi-scale training and testing.

---

## 9. Datasets, annotation & benchmarks

Common datasets:

- **PASCAL VOC:** ~20 classes; used for early detectors.
- **MS COCO:** 80 classes; large, diverse, known for AP@[.5:.95] evaluation.
- **Open Images:** very large, many classes, complex annotations.
- Domain-specific datasets exist for driving (KITTI, BDD100K), aerial, medical, retail.

Annotation: bounding boxes, sometimes instance masks (COCO has both), and difficult/ignore flags. Quality of annotations is critical — label noise harms training.

---

## 10. Training recipes & augmentation specific to detection

- **Pretrain backbone** on ImageNet; fine-tune for detection.
- **Batch size & learning rate schedule:** often use SGD with momentum and step decay or cosine.
- **Warmup** learning rate for a few iterations to stabilize early training.
- **Augmentation:** random horizontal flip, photometric distortions, random crop and scale, mosaic augmentation (YOLOv4), MixUp/CutMix variants adapted to boxes, multi-scale training (randomly resize input).
- **Anchor box design:** use k-means clustering on dataset box sizes to choose anchor scales/aspect ratios (as in YOLOv2).
- **Normalization:** SyncBatchNorm across GPUs if per-GPU batch size small.
- **Hard example mining:** OHEM or focal loss.

---

## 11. Practical implementation details & pitfalls

- **Box parameterization** must match the loss used; wrong encoding leads to training instability.
- **Bounding box clipping** to image borders prevents degenerate boxes.
- **Imbalanced classes**: use class re-weighting, focal loss, or oversampling for rare classes.
- **IoU thresholds for positive/negative** selection are important (e.g., RPN uses 0.7 pos, 0.3 neg in original Faster R-CNN).
- **Evaluation protocol** must match benchmark specs (COCO vs VOC use different AP definitions).
- **Confidence calibration:** predicted scores must be reliable; temperature scaling or score calibration may be applied.
- **Anchors explosion:** too many anchors (scales/aspect ratios) increase computation and imbalance.

---

## 12. Advanced topics and modern improvements

### Feature fusion and attention

- Attention modules (SE, CBAM) and Transformer-style cross-attention enhance feature representation.

### EfficientDet & BiFPN

- Compound scaling and BiFPN achieve good trade-off between accuracy and efficiency using weighted feature fusion.

### Anchor-free advances

- **FCOS, CenterNet, TTFNet** remove anchor hyperparameters and simplify matching. They often predict center/offset/size and use centerness/heatmaps.

### Cascade & multi-stage heads

- **Cascade R-CNN** increases IoU thresholds in successive stages for progressively refined boxes and better localization.

### Mask & Keypoint heads

- Instance segmentation (Mask R-CNN) adds a mask head on top of the detection pipeline.
- Pose/keypoint detection often built as extensions of detection with heatmap outputs.

### Self-supervised & weakly-supervised detection

- Pretraining with self-supervised tasks and using weak labels reduce labeling effort.

---

## 13. Deployment: speed vs accuracy trade-offs

- **Real-time detectors** (e.g., YOLO family, SSD-lite, MobileNet-SSD) trade accuracy for speed; use small backbones, fewer anchors, and optimized heads.
- **Quantization & pruning**: reduce model size and latency for edge devices.
- **NMS and post-processing** should be GPU-optimized for throughput; batching inference requires careful I/O and NMS handling.
- **TTA (Test Time Augmentation)** can boost accuracy but increases latency.

---

## 14. Example math snippets & formulas

### Box encoding (common)

Given anchor (a) with center ((x*a, y_a)) and size ((w_a, h_a)), GT box ((x*{gt}, y*{gt}, w*{gt}, h*{gt})):
[
t_x = (x*{gt} - x*a) / w_a,\quad t_y = (y*{gt} - y*a) / h_a
]
[
t_w = \log(w*{gt}/w*a),\quad t_h = \log(h*{gt}/h_a)
]
Predicted offsets (\hat{t}) are regressed and inverted at inference.

### Focal Loss (binary)

[
FL(p_t) = -\alpha(1-p_t)^\gamma \log(p_t)
]
Use (\gamma\in[0,3]) typically; (\alpha) balances positive/negative.

### Smooth L1 loss

See section 6 (used for bbox regression).

---

## 15. Troubleshooting & best practices

- If many false positives at low scores: increase confidence threshold, improve NMS or use soft-NMS.
- If low recall (missed objects): ensure anchors cover box sizes, increase proposals (in two-stage), use multi-scale or FPN.
- If localization poor: use IoU/GIoU/DIoU losses, cascade heads, or better anchor design.
- If training unstable: reduce learning rate, use warmup, verify label/anchor matching, clip gradients.
- If overfitting: use stronger augmentation, weight decay, or reduce model capacity.

---

## 16. Relation to other tasks

- **Instance segmentation** extends detection by predicting pixel masks per detected instance (Mask R-CNN).
- **Tracking (MOT)** connects detections across frames—reliable detection is critical to tracking performance.
- **Visual grounding / referring expression** links textual mentions to detected boxes.

---

## 17. Summary

- Object detection = localization + classification — solved by a wide spectrum of architectures: two-stage (accurate), one-stage (fast), anchor-based (flexible), anchor-free (simple).
- Core challenges: class imbalance, small-object detection, multi-scale handling, and accurate localization.
- Successful detectors combine a powerful backbone, multi-scale feature fusion (FPN), carefully designed head, good loss design (focal, IoU variants), robust augmentation, and appropriate post-processing (NMS/soft-NMS).
- Real-world deployment requires trading accuracy for speed, model compression, and careful engineering of inference pipelines.

---
