# Atlas data dictionary

## Canopy COGs

Each city file is a single-band `uint8` Cloud Optimized GeoTIFF in EPSG:4326.

| Value | Meaning |
|---:|---|
| 0 | Pixel classified as non-canopy |
| 255 | Pixel classified as visible woody canopy |
| 254 | Nodata outside the valid mapped footprint |

The native grid is approximately 0.6 m. Internal overviews use average
resampling only to support efficient low-zoom display. They do not replace the
native binary observations. From zoom 15, the tile API reads the native grid
with a five-pixel halo and suppresses 8-connected canopy components containing
only 1-5 pixels. The halo makes the result independent of web-tile boundaries.
Zooms 15-16 are then average-resampled; zoom 17 and above uses nearest-neighbour
rendering. Point queries apply the same native-pixel rule. This filter affects
only atlas display and inspection: the archived COG values and all frozen city
and neighbourhood statistics remain unchanged.

## City catalogue

| Field | Definition |
|---|---|
| `rank` | Population rank in the frozen 50-city registry |
| `slug` | Stable URL-safe city identifier |
| `cityKey` | Analysis-pipeline city identifier |
| `bounds` | West, south, east and north raster or boundary extent |
| `center` | Display centre derived from `bounds` |
| `areaWeightedCanopyPct` | Canopy area divided by mapped city area, percent |
| `populationWeightedCanopyPct` | Block-group canopy context weighted by residents, percent |
| `population` | Population represented in the descriptive city frame |
| `tracts` | Number of fully contained census tracts in the frozen 11,984-unit Result 2 tract frame |
| `blockGroups` | Number of municipal block-group intersections in the canopy atlas |
| `imagery` | Earliest, median and latest acquisition dates and unique date count |
| `raster.available` | Whether the release COG was present when the catalogue was built |

## Neighbourhood evidence

The evidence layer contains 34,566 fully contained 2020 census block groups in
50 cities from evidence release `GHV27-2026-08-09`. Each city is stored as
one GeoJSON feature collection. The public property names are shortened to keep
the interactive files compact.

| Field | Definition |
|---|---|
| `geoid` | Twelve-digit census block-group GEOID |
| `pop` | Represented ACS population |
| `canopy` | Analysis-frame canopy cover, percent |
| `g` | Non-negative canopy deficit from a density-comparable city q75 reference, percentage points |
| `h` | Positive within-city summer LST anomaly, degrees C; negative anomalies are structurally zero |
| `v` | City-balanced social-condition ordering on [0, 1], built from four ACS indicators in three equally weighted domains |
| `gl`, `hl`, `vl` | Frozen three-level classes for G, H and V |
| `type` | One of 27 mutually exclusive nominal labels, `GxHyVz` |
| `context` | One of six manuscript-facing combinations, A-F |
| `a_u`, `a_g`, `a_h`, `a_v` | Fraction of local analytical capacity selected at the city-specific 10% budget under uniform, canopy-deficit-first, surface-heat-first and social-condition-first ranking |

The social-condition ordering averages three domains: poverty; the mean of
renter occupancy and households without vehicles; and the share aged 65 years
or older. Race and ethnicity do not enter V. The 27 types are nominal
combinations, not an ordered risk score.

### Six displayed contexts

| Context | Frozen definition |
|---|---|
| A | G2/G3, H3 and V3: joint canopy-deficit, surface-heat and social context |
| B | G2/G3, H3 and V1/V2: surface-heat canopy deficit |
| C | G2/G3, H1/H2 and V3: social-condition canopy deficit |
| D | G2/G3, H1/H2 and V1/V2: canopy-deficit dominant |
| E | G1 with H3 or V3: heat or social context with no measured canopy deficit |
| F | G1, H1/H2 and V1/V2: lower or mixed reference |

`G3H3V3` is the strict triple-high subset of context A. Context A must not be
described as though every member were strict triple-high.

## Interpretation boundary

The mapped class is visible woody crown in the source aerial imagery. It is not
the footprint of cast shade at a particular time, an inventory of individual
trees, a measure of personal exposure, or a parcel-level assessment of where
planting is feasible. LST is a surface-temperature measure rather than personal
heat exposure. The allocation layer is a controlled analytical comparison that
changes ranking under an equal area budget; it is not a planting plan or a
prediction of realised cooling.
