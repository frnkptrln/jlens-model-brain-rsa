# Preregistered analysis plan

## Title

**Does Jacobian transport improve alignment between language-model and visual-cortical representational geometry?**

Status: frozen Stage 1 plan, 15 August 2026, before inspection of any
J-lens-to-fMRI association.

## Scope and claim boundary

This study tests a representational association. It does not test whether the
human brain literally implements a J-space, whether a participant was
conscious of a representation, or whether the compared systems use the same
features or computations. The primary model variable is Jacobian transport,
not Anthropic's sparse non-negative J-space projection.

## Data

- Natural Scenes Dataset (NSD), eight participants.
- The 515 images that occur exactly three times for every participant.
- Trial responses are averaged within image using the preparation of Doerig et
  al.; neural inputs are the `special515` condensed ROI RDMs produced either
  by their compact cache or by an exact reconstruction from the public
  fsaverage beta files.
- COCO captions are obtained from the public 2017 annotation archive and
  joined through `nsd_stim_info_merged.csv`.
- The frozen preparation currently yields 515 images and 2,578 captions: 512
  images have five annotations and three have six. All available annotations
  are averaged, matching the behavior of the public embedding code.
- No NSD image pixels are used by the text models.

## Model and readout

- Confirmatory model: `Qwen/Qwen3-4B`, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- This is a text-only causal language model. A multimodally trained checkpoint
  is excluded to avoid visual-pretraining confounds.
- Jacobian lens: `neuronpedia/jacobian-lens`, revision
  `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a`, file
  `qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt`.
- Reference implementation: `anthropics/jacobian-lens`, commit
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- No text generation is performed.
- Readout position: final attended caption token after removing a terminal
  special token, if present.
- Sampled block indices: 9, 18, 27, and 34 of 36 blocks.

For residual state \(h_l\), the two matched representations are:

\[
r_l^{\mathrm{raw}} = \operatorname{norm}(h_l), \qquad
r_l^{J} = \operatorname{norm}(J_l h_l).
\]

Both have identical dimensionality. They differ only by the fixed
corpus-averaged Jacobian transport. Caption vectors are averaged within image
after extraction.

## Prompt conditions

- Confirmatory: the raw COCO caption and no chat template.
- Exploratory: `Imagine the visual scene described by this caption: <caption>`.

The imagery instruction is not treated as neutral preprocessing because the
J-space paper predicts and demonstrates instruction-dependent workspace
contents.

## Neural regions

The confirmatory high-level family comprises the ventral, lateral, and
parietal ROIs from the NSD `streams` atlas. Before model-brain scores are
opened, the literal cache labels corresponding to those three atlas entries
will be recorded in the run manifest. EVC is a specificity/control ROI and is
not part of the aggregate primary test.

## Representational similarity analysis

For each model representation and each participant/ROI:

1. compute a condensed image-by-image RDM using correlation distance;
2. correlate model and neural RDMs using Pearson correlation, matching the
   principal Doerig et al. analysis;
3. retain one RSA value per participant, ROI, layer, prompt, and
   representation.

The image-pair entries in an RDM are not treated as independent observations.
The inferential unit is the participant.

## Confirmatory hypothesis and test

Directional hypothesis: Jacobian transport increases alignment in high-level
visual cortex at workspace-candidate middle layers.

For each participant, calculate

\[
\Delta_s = \operatorname{mean}_{r\in\{ventral,lateral,parietal\},
l\in\{18,27\}}
\left( RSA^J_{srl} - RSA^{raw}_{srl} \right).
\]

The single primary test is an exact two-sided sign-flip test of \(\Delta_s\)
around zero across eight participants. A two-sided test is retained despite
the directional prediction. Report the mean difference, participant-level
standard error, Cohen's \(d_z\), exact p-value, and the paired t-test only as a
parametric companion. Alpha is 0.05.

With eight participants, the smallest attainable two-sided exact sign-flip
p-value is 2/256 = 0.0078125. The fixed dataset size cannot be increased by
counting the 132,355 image pairs as independent samples.

## Secondary and exploratory analyses

- Component-wise raw-versus-J contrasts for every layer × ROI in the raw
  prompt condition. Benjamini-Hochberg FDR is applied across that component
  family.
- Layer profile across indices 9, 18, 27, and 34.
- EVC specificity control.
- Raw-caption versus imagery-instruction comparison. This entire prompt
  analysis is exploratory and receives its own FDR family.
- MPNet (`all-mpnet-base-v2`, pinned revision
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`) is a descriptive external
  benchmark. It is not included in the causal-within-model J-versus-raw
  contrast.
- Leave-one-participant-out neural noise ceilings and noise-ceiling-normalized
  scores are descriptive; raw RSA is the inferential target.
- Caption-level reliability is descriptive.

Encoding models, searchlights, sparse J-space projection, model generation,
and additional model families are excluded from the confirmatory pilot.

## Quality control and exclusions

- Abort rather than silently continue if there are not exactly 515 shared
  three-repeat NSD IDs.
- Abort on duplicate/missing IDs, non-finite representations or RDMs, constant
  feature rows, incompatible RDM lengths, mismatched model/lens widths, or a
  requested layer absent from the fitted lens.
- Include all eight participants when their complete shared-515 RDMs are
  available. Any exclusion must be made without consulting model-brain scores
  and entered below.
- Do not choose layers, ROIs, prompt wording, distance metrics, or correlation
  types after viewing the confirmatory results.

## Outcome interpretation

- Positive primary result: evidence that a fixed Jacobian transport improves
  representational alignment relative to the same model's raw residuals under
  this task and dataset.
- Null result: no evidence for improvement at the tested layers/ROIs; it does
  not establish equivalence or refute all J-space/brain relationships.
- Negative result: evidence that the transport reduces alignment under this
  operationalization.
- No outcome licenses the claim that the brain possesses a J-space.

## Deviations log

No deviations at freeze time.

- **15 August 2026, before any model-to-brain score was available:** the
  deadline-constrained confirmatory run extracts only the raw-caption condition
  at preregistered layers 18 and 27. Layers 9 and 34 and the imagery prompt are
  deferred exploratory analyses. This does not change the primary contrast,
  which was defined only over the raw prompt and layers 18 and 27. The reason is
  bounded remote-result persistence and turnaround time, not observed neural
  alignment.
- **15 August 2026, before any model-to-brain score was available:** because the
  group's compact neural cache had not yet been supplied, the same
  `special515` neural RDMs are reconstructed from the public fsaverage
  `betas_fithrf_GLMdenoise_RR` files. The reconstruction retains the published
  session-wise vertex z-scoring, three-repeat averaging, `streams` masks,
  correlation distance, stimulus order, and ROI labels. This changes the data
  access route, not the neural estimand or confirmatory test.

Every later entry must include date, change, reason, whether scores had been
viewed, and whether the affected analysis is relabelled exploratory.
