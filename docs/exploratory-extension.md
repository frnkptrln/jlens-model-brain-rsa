# Prospective plan for the post-pilot exploratory extension

Date frozen: 15 August 2026, after the primary raw-caption layer-18/layer-27
result was known and before any new model extractions were launched.

This document is not a preregistration of the completed pilot. It fixes the
analysis boundary for six previously uncomputed prompt-by-layer conditions.

## Question

Does the effect of dense Jacobian transport on NSD brain alignment depend on
model depth or on an explicit visual-imagery instruction?

## Fixed design

- Model and lens: the same pinned Qwen3-4B checkpoint and public fitted lens as
  in the completed pilot.
- Stimuli: the same 2,578 captions pooled within the same 515 shared NSD images.
- Readout: the same final attended non-terminal-special token.
- Representations: `norm(h_l)` and `norm(J_l h_l)`; this is not formal sparse
  non-negative J-space.
- Layers: 9, 18, 27, and 34.
- Prompts: the unmodified caption (`raw`) and
  `Imagine the visual scene described by this caption: {caption}` (`imagine`).
- Neural targets: the already reconstructed ventral, lateral, and parietal
  participant-level RDMs for all eight NSD participants.
- RSA: Pearson correlation between condensed correlation-distance RDMs.

The raw-caption layer-18 and layer-27 cells have already been observed. They
will be shown for continuity but excluded from any claim that relies on this
prospective extension. The six new cells are raw layers 9 and 34 and imagine
layers 9, 18, 27, and 34.

## Estimands and inference

1. Report participant-level and group-mean `J-lens minus raw` RSA for every new
   prompt-by-layer-by-ROI cell.
2. Apply exact two-sided participant sign-flip tests. Control Benjamini-Hochberg
   FDR across the 18 new cells only.
3. Report layer trajectories descriptively. No directional imagery-prompt
   hypothesis is asserted after seeing the original result.
4. Estimate prompt moderation at each layer as
   `(imagine J-raw delta) - (raw J-raw delta)` and label it exploratory.
5. Report raw-versus-J-lens model-RDM correlations for each condition to
   distinguish geometric similarity from similar brain-RSA scores.

No voxel, image, ROI, layer, prompt, or model will be selected post hoc to
redefine a primary result. Any additional model is a separate replication and
will not be pooled silently with Qwen3-4B.

## Claim boundary

The extension can establish configuration dependence or robustness of a
representational association. It cannot establish that the brain implements
J-space, shares the same features or computation, or contains a homologous
global workspace.
