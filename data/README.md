# Supporting data

The minimum supporting dataset for this repository is published on Zenodo:

**https://doi.org/10.5281/zenodo.21976809**

Version 1.0.0 contains frozen block-group and tract analysis-unit ledgers,
source data for five main figures, seventeen Supplementary figures and eight
Supplementary tables, compact model and bootstrap outputs, random seeds,
quality-control reports, checksums and licence-permitted neighbourhood-level
derived spatial data.

This Git directory intentionally contains no duplicate analysis archive.
Lightweight neighbourhood GeoJSON layers required by the interactive atlas are
tracked under `atlas/public/neighbourhoods/` and documented in
`atlas/docs/data_dictionary.md`.

The Zenodo record is a minimum supporting dataset rather than a raw-input or
end-to-end reconstruction archive. Raw high-resolution imagery,
full-resolution 0.6 m canopy masks, third-party raster downloads,
model-development patches and checkpoints, temporary caches and superseded
analyses are excluded. See `../docs/data_contract.md` and
`../docs/reproducibility_scope.md` for the exact boundary. Do not place private
data, raw licensed imagery or model checkpoints in this Git repository.
