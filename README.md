# J-Lens × NSD RSA Experiment

Reproducibility repository for a targeted-region experiment conducted during
the Digital Minds Research Sprint (August 2026).

## Research question

Does dense Jacobian transport change the representational similarity between
language-model caption representations and fMRI response geometry from the
Natural Scenes Dataset (NSD)?

The experiment compares matched representations from `Qwen/Qwen3-4B`:

```text
raw:       norm(h_l)
transport: norm(J_l h_l)
```

The confirmatory analysis used 515 NSD images shared by all eight participants,
2,578 COCO captions, layers 18 and 27, and bilateral ventral, lateral, and
parietal higher-visual ROIs. The inferential unit was the participant.

## Main result

The preregistered aggregate RSA was lower after dense Jacobian transport:

- raw mean RSA: `0.09782`
- transported mean RSA: `0.08558`
- mean difference: `-0.01224`
- exact two-sided sign-flip p: `0.0078125`
- participant direction: `8/8` negative

The frozen extension found that the sign and magnitude of the contrast varied
with layer, prompt, and final normalization. The supported conclusion is
therefore specific: dense Jacobian transport changes model-brain geometry under
this operationalization, but not in a measurement-invariant direction.

This does **not** show that the brain implements J-space. It also does not test
Anthropic's formal sparse non-negative J-space projection; the tested quantity
is the dense linear transport `J_l h_l`.

## Repository map

| Path | Contents |
| --- | --- |
| `src/jspace_fmri/` | Data preparation, extraction, RSA, statistics, and diagnostics |
| `configs/` | Frozen confirmatory and smoke-test configurations |
| `tests/` | Unit and regression tests |
| `results/confirmatory/` | Preregistered participant- and group-level results |
| `results/extension/` | Exploratory layer, prompt, and normalization summaries |
| `results/diagnostics/` | Compact Jacobian-spectrum diagnostic |
| `docs/` | Preregistration, deviations, diagnostics, and extension notes |
| `provenance/` | Validation manifest with hashes and pinned revisions |

The separate whole-cortex experiment that motivated the later cross-study
comparison remains in its own repository:
[`lucas-nunn/natural-j-scenes`](https://github.com/lucas-nunn/natural-j-scenes/tree/8389a3897a29c043bf12462b804400e999baa330).
Its paper and result files are not duplicated here.

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

The principal command-line stages expose their arguments through `--help`:

```bash
jspace-prepare-nsd --help
jspace-raw-nsd --help
jspace-extract --help
jspace-rsa --help
jspace-extension --help
```

Regenerate the experiment-specific figures with:

```bash
python scripts/make_primary_figure.py \
  --analysis-dir results/confirmatory \
  --output results/confirmatory/primary_delta.png

python scripts/make_extension_figure.py \
  --table results/extension/layer_trajectory_group.csv \
  --output results/extension/layer_trajectory.png
```

## Data availability

No NSD images, raw fMRI files, response arrays, captions, credentials, or large
intermediate model RDMs are redistributed. Obtain NSD data under the dataset's
access terms from [naturalscenesdataset.org](https://naturalscenesdataset.org/).
The expected local schemas and conversion routes are documented in
[`data/README.md`](data/README.md).

The committed files are compact derived summaries sufficient to inspect the
reported statistics and regenerate the experiment figures. Full recomputation
requires authorized NSD access and the pinned model/lens artifacts recorded in
[`docs/preregistration.md`](docs/preregistration.md) and
[`provenance/validation-manifest.json`](provenance/validation-manifest.json).

## Reproducibility status

- All 16 tests pass on the publication bundle.
- Model, lens, and implementation revisions are pinned.
- The confirmatory plan and dated deviations log are preserved.
- Unfinished coordinate-control jobs are not included or presented as evidence.

## License and citation

Code is released under the MIT License. See [`CITATION.cff`](CITATION.cff) for
software citation metadata. Dataset and model artifacts remain subject to their
own licenses and access conditions.
