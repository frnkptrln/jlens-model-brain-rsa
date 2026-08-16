# Dense-transport diagnostic plan

Date frozen: 15 August 2026, after the completed layer-18/layer-27 pilot and
before these diagnostics were launched.

These analyses diagnose the control representation `J_l h_l`; they do not turn
it into formal sparse J-space.

## Jacobian spectrum

For fitted Jacobians at layers 9, 18, 27, and 34, retain the complete singular
value spectrum and report:

- maximum and minimum singular value;
- condition number where numerically defined;
- rank counts at relative thresholds `1e-2`, `1e-3`, `1e-4`, and `1e-6` of the
  largest singular value;
- stable rank and entropy-based effective rank;
- cumulative squared-spectrum energy at 32, 64, 128, 256, 512, and 1024
  components.

This distinguishes severe rank loss from a full-rank but strongly anisotropic
reweighting. Invertibility alone will not be described as geometry
preservation: a non-orthogonal invertible map can substantially change an RDM.

## Normalization control

Recompute raw-caption layer-18 and layer-27 correlation-distance RDMs for
`h_l` and `J_l h_l` without the model's learned final RMSNorm. Compare their
brain RSA with the already observed full-RMSNorm result.

A gain-free RMS normalization multiplies each stimulus vector by a positive
scalar. Correlation distance is invariant to that operation, apart from
numerical epsilon effects, so it is analytically equivalent to the no-norm RDM
and will not be run as a redundant condition. The learned per-dimension gain is
the geometry-changing component tested by this control.

## EVC control

The early-stream ROI (stream-mask ID 1) will be added only from a verified
compact cache or a separately documented reconstruction. It is not available
in the current saved neural artifacts, and will not be approximated from the
three higher-level ROIs.

## Interpretation

If the dense map is rank-deficient, information loss is a plausible mediator
of any reduced alignment. If it is full-rank but anisotropic, the result is
better described as a change in representational geometry. If removal of the
learned RMSNorm gain eliminates the effect, the result is a normalization
interaction rather than evidence about workspace-selective directions.
