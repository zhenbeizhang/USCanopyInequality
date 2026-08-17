# Code-release checklist

- [x] Freeze a semantic repository structure for Results 1-5, figures and QA.
- [x] Remove workstation-specific absolute paths from Python string literals.
- [x] Exclude research data, imagery, checkpoints and manuscript files.
- [x] Provide Conda and pip environment specifications.
- [x] Provide a staged command-line runner and data contract.
- [x] Record script-level and file-level manifests with SHA-256 checksums.
- [x] Strictly load the frozen epoch-495 checkpoint with the public model class.
- [ ] Run the complete pipeline against the final deposited data record.
- [ ] Add all software contributors and ORCID identifiers to CITATION.cff.
- [ ] Create a tagged GitHub release and archive that exact tag for a DOI.
- [ ] Add the software DOI and separate data DOI to the manuscript.
- [ ] Verify licences and access statements for every redistributed data file.
- [ ] Replace provisional availability text with issued persistent identifiers.

The unchecked items require the future data deposit, contributor metadata or a
repository-issued identifier. They should not be marked complete by assumption.
