# Mind the Gap

**Measurement Dependence in the Brain Alignment of Jacobian-Transported Language Representations**

Companion repository for the Digital Minds Research Sprint paper by Lucas Nunn,
Anke Borchers, and Frank Peterlein (August 2026).

## What this repository tests

We ask whether applying a learned Jacobian transport to language-model residual
representations changes their representational similarity to fMRI responses from
the Natural Scenes Dataset (NSD). Two complementary pipelines produced different
answers:

- **Study A — whole-cortex searchlight:** late Qwen3.5-4B layers showed small
  positive changes that were spatially non-uniform but did not isolate a
  candidate workspace network.
- **Study B — targeted ROI analysis:** the preregistered Qwen3-4B aggregate over
  layers 18 and 27 and ventral, lateral, and parietal ROIs was lower after
  transport (mean delta = -0.01224; exact two-sided sign-flip p = 0.0078125;
  8/8 participants negative).
- **Study B extension:** the sign and size of the contrast varied with layer,
  prompt, and normalization, showing that the effect is not pipeline-invariant.

The scientifically defensible conclusion is therefore methodological: brain
alignment for Jacobian-transported representations depends on the measurement
pipeline. These results do **not** show that the brain implements J-space, and
they do not test Anthropic's formal sparse non-negative J-space projection. The
tested quantity is the dense linear transport `J_l h_l`.

## Repository map

| Path | Contents |
| --- | --- |
| `src/jspace_fmri/` | Study B data preparation, extraction, RSA, statistics, and diagnostics |
| `configs/` | Frozen model and analysis configurations |
| `tests/` | Unit and regression tests for the Study B pipeline |
| `results/study_a/` | Compact public summary of Study A |
| `results/study_b/` | Compact participant/group summaries and reported diagnostics |
| `paper/` | Final sprint paper, figure, preregistration, and analysis notes |
| `provenance/` | Validation manifest with hashes and pinned revisions |

Study A's full whole-cortex implementation is maintained in
[`lucas-nunn/natural-j-scenes`](https://github.com/lucas-nunn/natural-j-scenes)
and is pinned here to commit
[`8389a3897a29c043bf12462b804400e999baa330`](https://github.com/lucas-nunn/natural-j-scenes/tree/8389a3897a29c043bf12462b804400e999baa330).

## Installation and checks

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
```

Model extraction additionally requires the pinned model extras:

```bash
pip install -e ".[model]"
```

All command-line entry points expose their exact arguments through `--help`.
The main stages are:

```bash
jspace-prepare-nsd --help
jspace-raw-nsd --help
jspace-extract --help
jspace-rsa --help
jspace-extension --help
```

Regenerate the combined paper figure from the committed compact tables with:

```bash
python scripts/make_paper_figure.py
```

## Data availability

No NSD images, raw fMRI files, response arrays, captions, access credentials, or
large intermediate model RDMs are redistributed. Obtain NSD data under the
dataset's access terms from [naturalscenesdataset.org](https://naturalscenesdataset.org/).
The expected local schemas and conversion routes are documented in
[`data/README.md`](data/README.md).

The committed result files are compact derived summaries sufficient to inspect
the reported statistics and regenerate the paper figure. Full recomputation
requires authorized NSD access and the pinned model/lens artifacts listed in the
preregistration and provenance manifest.

## Reproducibility status

- Study B tests pass on the publication bundle.
- Model, lens, implementation, and Study A revisions are pinned.
- The confirmatory plan and its dated deviations log are preserved.
- Unfinished coordinate-control jobs are not presented as completed evidence.

## License and citation

Code is released under the MIT License. See [`CITATION.cff`](CITATION.cff) for
citation metadata. Dataset and model artifacts remain subject to their own
licenses and access conditions.
