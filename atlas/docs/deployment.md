# Deployment design

## Release boundary

The public map and the archived dataset serve different purposes. The map is an
inspection interface; the archive is the citable record supporting the
reported analyses. The current release provides both:

1. Zenodo record https://doi.org/10.5281/zenodo.21976810, containing frozen
   analysis-unit ledgers, panel-level source data, compact model and bootstrap
   outputs, metadata, checksums and neighbourhood-level derived spatial data.
2. A public web application that displays the same frozen neighbourhood layer
   and provides on-demand visual access to the 0.6 m canopy layer.

Full-resolution masks and COGs are deployment assets and are not part of the
minimum Zenodo dataset. The interface is therefore not an archival route to
those rasters. All numerical claims remain inspectable through the aggregated
and panel-level files in the DOI record.

## Recommended production layout

- Static React build and the 50 compact neighbourhood GeoJSON files on a
  conventional static host or CDN with compression enabled.
- FastAPI service on a container platform.
- COGs in S3, R2 or another object store supporting HTTP byte-range requests.
- A public OpenFreeMap style as the contextual basemap.
- HTTPS and a stable project subdomain.

Set the API environment as follows:

```text
ATLAS_CATALOG=/app/public/catalog/cities.json
ATLAS_COG_BASE_URL=https://data.example.org/canopy/v1
ATLAS_ALLOWED_ORIGINS=https://canopy.example.org
```

The API accepts only city slugs in the frozen catalogue and never exposes an
arbitrary file or URL reader. If the web client and API use different origins,
set `ATLAS_ALLOWED_ORIGINS` to the final web origin.

## Rendering semantics

- Below zoom 17, average resampling encodes canopy fraction in alpha.
- At zoom 17 and above, nearest-neighbour resampling preserves the binary mask.
- Nodata remains transparent.
- Point queries return `canopy`, `not_canopy` or `nodata`.

These semantics must remain fixed within a release. Any threshold, colour or
zoom transition change requires a new application version and a short release
note.

## Pre-release checks

1. Run `scripts/audit_atlas.py --require-all`.
2. Run `scripts/audit_neighbourhood_layers.py` and confirm the frozen 50-city,
   34,566-feature release has no issues.
3. Confirm the 50 neighbourhood-file SHA-256 values against the public release
   manifest and verify the server-side COGs against the internal deployment
   manifest.
4. Test national selection and at least one dense, sparse, large and coastal
   city at desktop and mobile widths.
5. Check zooms 8, 12, 17 and 20 for nonblank canopy tiles and correct binary
   behaviour.
6. Confirm share links restore the selected city, evidence view, context filter
   and allocation rule.
7. Verify G/H/V labels, the six-context counts and selected-neighbourhood
   properties against the frozen source data.
8. Verify that the About dialog states the scientific boundaries.
9. Record the deployed application commit, public atlas URL and supporting-data
   DOI in the manuscript.
