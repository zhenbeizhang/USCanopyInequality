import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronLeft,
  CircleHelp,
  Copy,
  Layers3,
  LocateFixed,
  MapPinned,
  Menu,
  Search,
  Trees,
  X,
} from "lucide-react";
import {
  ALLOCATION_LABELS,
  ALLOCATION_PROPERTIES,
  CONTEXT_META,
  OVERLAY_LABELS,
  contextTitle,
  levelLabel,
  typeCellColor,
} from "./evidence";
import type {
  AllocationRule,
  City,
  ContextFilter,
  ContextId,
  NeighbourhoodOverlay,
  NeighbourhoodProperties,
  NeighbourhoodSummary,
  NeighbourhoodSummaryCatalog,
  PointInspection,
} from "./types";

type PanelMode = "cities" | "layers";

const MapView = lazy(() => import("./MapView"));

const OVERLAYS = Object.keys(OVERLAY_LABELS) as NeighbourhoodOverlay[];
const ALLOCATION_RULES = Object.keys(ALLOCATION_LABELS) as AllocationRule[];
const CONTEXT_IDS = Object.keys(CONTEXT_META) as ContextId[];

function formatPopulation(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

function formatRatio(value: number | null) {
  return value === null || !Number.isFinite(value) ? "Not estimable" : `${value.toFixed(2)}×`;
}

function classificationLabel(value: PointInspection["classification"]) {
  if (value === "canopy") return "Mapped as tree canopy";
  if (value === "not_canopy") return "Not mapped as tree canopy";
  return "No valid canopy data at this location";
}

function initialCitySlug() {
  return new URLSearchParams(window.location.search).get("city");
}

function initialOverlay(): NeighbourhoodOverlay {
  const value = new URLSearchParams(window.location.search).get("view") as NeighbourhoodOverlay | null;
  return value && OVERLAYS.includes(value) ? value : OVERLAYS[0];
}

function initialContextFilter(): ContextFilter {
  const value = new URLSearchParams(window.location.search).get("filter");
  if (value === "all" || value === "strict") return value;
  if (CONTEXT_IDS.includes(value as ContextId)) return value as ContextId;
  if (value && /^G[123]H[123]V[123]$/.test(value)) return value as ContextFilter;
  return "all";
}

function initialAllocationRule(): AllocationRule {
  const value = new URLSearchParams(window.location.search).get("rule") as AllocationRule | null;
  return value && ALLOCATION_RULES.includes(value) ? value : ALLOCATION_RULES[0];
}

function isCompactViewport() {
  return window.matchMedia("(max-width: 760px)").matches;
}

function ContextSummary({ summary }: { summary: NeighbourhoodSummary }) {
  return (
    <div className="context-summary">
      <div>
        <span>Joint context</span>
        <strong>{formatPopulation(summary.jointContext.neighbourhoods)}</strong>
        <small>{formatPopulation(summary.jointContext.population)} residents</small>
      </div>
      <div>
        <span>Strict G3H3V3</span>
        <strong>{formatPopulation(summary.strictTripleHigh.neighbourhoods)}</strong>
        <small>{formatPopulation(summary.strictTripleHigh.population)} residents</small>
      </div>
      <p>
        City point ratio to leave-one-out marginal expectation: {formatRatio(summary.jointContext.observedExpectedRatio)}
      </p>
    </div>
  );
}

function TypeExplorer({
  filter,
  onChange,
}: {
  filter: ContextFilter;
  onChange: (filter: ContextFilter) => void;
}) {
  const currentType = filter.startsWith("G") ? filter : null;
  const initialG = currentType ? Number(currentType[1]) : 3;
  const [gLevel, setGLevel] = useState(initialG);
  const hLevels = [3, 2, 1] as const;
  const vLevels = [1, 2, 3] as const;

  return (
    <details className="type-explorer">
      <summary>Explore all 27 nominal types</summary>
      <div className="g-level-tabs" aria-label="Canopy-deficit level">
        {[1, 2, 3].map((level) => (
          <button
            key={level}
            className={gLevel === level ? `active level-${level}` : `level-${level}`}
            onClick={() => setGLevel(level)}
            type="button"
          >
            G{level}
          </button>
        ))}
      </div>
      <div className="type-axis-label">Social-condition level</div>
      <div className="type-matrix">
        <span />
        {vLevels.map((level) => <b key={`v-${level}`}>V{level}</b>)}
        {hLevels.flatMap((hLevel) => [
          <b key={`h-${hLevel}`}>H{hLevel}</b>,
          ...vLevels.map((vLevel) => {
            const code = `G${gLevel}H${hLevel}V${vLevel}` as ContextFilter;
            const context = CONTEXT_IDS.find((id) => {
              if (id === "A") return gLevel >= 2 && hLevel === 3 && vLevel === 3;
              if (id === "B") return gLevel >= 2 && hLevel === 3 && vLevel < 3;
              if (id === "C") return gLevel >= 2 && hLevel < 3 && vLevel === 3;
              if (id === "D") return gLevel >= 2 && hLevel < 3 && vLevel < 3;
              if (id === "E") return gLevel === 1 && (hLevel === 3 || vLevel === 3);
              return gLevel === 1 && hLevel < 3 && vLevel < 3;
            }) ?? "F";
            return (
              <button
                key={code}
                className={filter === code ? "active" : ""}
                style={{ backgroundColor: typeCellColor(context, gLevel) }}
                title={`${code}: ${CONTEXT_META[context].title}`}
                aria-label={`${code}: ${CONTEXT_META[context].title}`}
                onClick={() => onChange(filter === code ? "all" : code)}
                type="button"
              >
                {hLevel}{vLevel}
              </button>
            );
          }),
        ])}
      </div>
      <p>Rows are H levels; columns are V levels. The selected G level is shown above.</p>
    </details>
  );
}

function DimensionLegend({ overlay }: { overlay: NeighbourhoodOverlay }) {
  const dimension = overlay === "canopy_gap" ? "G" : overlay === "surface_heat" ? "H" : "V";
  const classes = dimension === "G" ? "legend-g" : dimension === "H" ? "legend-h" : "legend-v";
  return (
    <div className="dimension-legend">
      {[1, 2, 3].map((level) => (
        <div key={level}>
          <i className={`${classes} level-${level}`} />
          <span>{levelLabel(dimension, level)}</span>
        </div>
      ))}
      <p>
        {dimension === "G" && "G is a canopy shortfall from a density-comparable q75 reference."}
        {dimension === "H" && "H is positive summer LST anomaly relative to the city reference."}
        {dimension === "V" && "V averages poverty, housing and mobility, and older-age domains, then orders neighbourhoods across cities."}
      </p>
    </div>
  );
}

function NeighbourhoodInspection({
  value,
  allocationRule,
}: {
  value: NeighbourhoodProperties | null;
  allocationRule: AllocationRule;
}) {
  if (!value) {
    return <p>Click a neighbourhood to inspect its three dimensions and assigned type.</p>;
  }
  const allocation = value[ALLOCATION_PROPERTIES[allocationRule]];
  return (
    <>
      <div className="inspection-title">
        <strong>{value.type}</strong>
        <span style={{ backgroundColor: CONTEXT_META[value.context].color }}>
          {CONTEXT_META[value.context].shortTitle}
        </span>
      </div>
      <div className="dimension-readout">
        <div>
          <b>G{value.gl}</b>
          <span>Canopy deficit</span>
          <strong>{value.g.toFixed(1)} pp</strong>
        </div>
        <div>
          <b>H{value.hl}</b>
          <span>Positive LST anomaly</span>
          <strong>{value.h.toFixed(1)} °C</strong>
        </div>
        <div>
          <b>V{value.vl}</b>
          <span>Social conditions</span>
          <strong>{Math.round(value.v * 100)}th pct.</strong>
        </div>
      </div>
      <dl className="neighbourhood-facts">
        <div><dt>Canopy cover</dt><dd>{value.canopy.toFixed(1)}%</dd></div>
        <div><dt>Represented population</dt><dd>{formatPopulation(value.pop)}</dd></div>
        <div><dt>GEOID</dt><dd>{value.geoid}</dd></div>
        <div>
          <dt>{ALLOCATION_LABELS[allocationRule]}</dt>
          <dd>{allocation > 0 ? `${Math.round(allocation * 100)}% of local capacity` : "Not allocated at 10%"}</dd>
        </div>
      </dl>
      <p className="inspection-note">
        {value.type === "G3H3V3"
          ? "This neighbourhood is in the strict triple-high subset."
          : CONTEXT_META[value.context].description}
      </p>
    </>
  );
}

export default function App() {
  const [cities, setCities] = useState<City[]>([]);
  const [summaries, setSummaries] = useState<NeighbourhoodSummaryCatalog | null>(null);
  const [selectedCity, setSelectedCity] = useState<City | null>(null);
  const [search, setSearch] = useState("");
  const [panelMode, setPanelMode] = useState<PanelMode>("cities");
  const [canopyVisible, setCanopyVisible] = useState(true);
  const [boundaryVisible, setBoundaryVisible] = useState(true);
  const [canopyOpacity, setCanopyOpacity] = useState(0.72);
  const [neighbourhoodOverlay, setNeighbourhoodOverlay] = useState(initialOverlay);
  const [neighbourhoodOpacity, setNeighbourhoodOpacity] = useState(0.67);
  const [contextFilter, setContextFilter] = useState<ContextFilter>(initialContextFilter);
  const [allocationRule, setAllocationRule] = useState(initialAllocationRule);
  const [inspection, setInspection] = useState<PointInspection | null>(null);
  const [neighbourhoodInspection, setNeighbourhoodInspection] = useState<NeighbourhoodProperties | null>(null);
  const [inspectionLoading, setInspectionLoading] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(() => !isCompactViewport());
  const [copied, setCopied] = useState(false);
  const requestedCitySlug = useRef(initialCitySlug());

  useEffect(() => {
    Promise.all([
      fetch(`${import.meta.env.BASE_URL}catalog/cities.json`).then((response) => {
        if (!response.ok) throw new Error("City catalogue unavailable");
        return response.json() as Promise<City[]>;
      }),
      fetch(`${import.meta.env.BASE_URL}neighbourhoods/summary.json`).then((response) => {
        if (!response.ok) throw new Error("Neighbourhood summary unavailable");
        return response.json() as Promise<NeighbourhoodSummaryCatalog>;
      }),
    ])
      .then(([items, summary]) => {
        setCities(items);
        setSummaries(summary);
        const requested = requestedCitySlug.current;
        if (requested) {
          const matched = items.find((city) => city.slug === requested) ?? null;
          setSelectedCity(matched);
          if (matched) setPanelMode("layers");
        }
      })
      .catch((error) => console.error(error));
  }, []);

  useEffect(() => {
    const url = new URL(window.location.href);
    if (selectedCity) {
      url.searchParams.set("city", selectedCity.slug);
      url.searchParams.set("view", neighbourhoodOverlay);
      if (neighbourhoodOverlay === "allocation") url.searchParams.set("rule", allocationRule);
      else url.searchParams.delete("rule");
      if (neighbourhoodOverlay === "contexts" && contextFilter !== "all") {
        url.searchParams.set("filter", contextFilter);
      } else {
        url.searchParams.delete("filter");
      }
    } else {
      url.searchParams.delete("city");
      url.searchParams.delete("view");
      url.searchParams.delete("rule");
      url.searchParams.delete("filter");
    }
    window.history.replaceState({}, "", url);
  }, [selectedCity, neighbourhoodOverlay, allocationRule, contextFilter]);

  useEffect(() => {
    setInspection(null);
    setNeighbourhoodInspection(null);
  }, [selectedCity]);

  const visibleCities = useMemo(() => {
    const needle = search.trim().toLowerCase();
    if (!needle) return cities;
    return cities.filter(
      (city) => city.name.toLowerCase().includes(needle) || city.state.toLowerCase().includes(needle),
    );
  }, [cities, search]);

  const selectedSummary = selectedCity && summaries ? summaries.cities[selectedCity.slug] : null;

  const handleSelectCity = useCallback((city: City) => {
    setSelectedCity(city);
    setNeighbourhoodOverlay(OVERLAYS[0]);
    setAllocationRule(ALLOCATION_RULES[0]);
    setContextFilter("all");
    setSidebarOpen(!isCompactViewport());
    setPanelMode("layers");
  }, []);

  const handleNationalView = useCallback(() => {
    setSelectedCity(null);
    setNeighbourhoodOverlay(OVERLAYS[0]);
    setAllocationRule(ALLOCATION_RULES[0]);
    setContextFilter("all");
    setPanelMode("cities");
  }, []);

  const handleInspect = useCallback((value: PointInspection | null, loading: boolean) => {
    setInspection(value);
    setInspectionLoading(loading);
  }, []);

  const handleInspectNeighbourhood = useCallback((value: NeighbourhoodProperties | null) => {
    setNeighbourhoodInspection(value);
  }, []);

  const handleOverlay = (value: NeighbourhoodOverlay) => {
    setNeighbourhoodOverlay(value);
    setNeighbourhoodInspection(null);
    setInspection(null);
  };

  const handleContextFilter = (value: ContextFilter) => {
    setContextFilter(value);
    setNeighbourhoodInspection(null);
  };

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };

  return (
    <main className="atlas-shell">
      <header className="topbar">
        <button
          className="icon-button mobile-only"
          onClick={() => setSidebarOpen((value) => !value)}
          title="Open map tools"
          aria-label="Open map tools"
        >
          <Menu size={20} />
        </button>
        <div className="brand">
          <Trees size={22} strokeWidth={1.8} />
          <div>
            <strong>US Urban Canopy Evidence Atlas</strong>
            <span>Canopy, neighbourhood contexts and allocation evidence</span>
          </div>
        </div>
        <div className="topbar-context">
          {selectedCity ? (
            <>
              <span>{selectedCity.name}, {selectedCity.state}</span>
              <button className="text-command" onClick={handleNationalView}>National view</button>
            </>
          ) : (
            <span>Select a city to inspect canopy and neighbourhood evidence</span>
          )}
        </div>
        <div className="topbar-actions">
          <span className="dataset-mark">50 cities | 34,566 classified neighbourhoods</span>
          <button className="icon-button" onClick={copyLink} title="Copy a link to this view" aria-label="Copy a link to this view">
            {copied ? <Check size={19} /> : <Copy size={19} />}
          </button>
          <button className="icon-button" onClick={() => setAboutOpen(true)} title="About the data" aria-label="About the data">
            <CircleHelp size={20} />
          </button>
        </div>
      </header>

      <div className="workspace">
        <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
          <div className="sidebar-tabs" role="tablist" aria-label="Map tools">
            <button className={panelMode === "cities" ? "active" : ""} onClick={handleNationalView} role="tab">
              <MapPinned size={17} />Cities
            </button>
            <button className={panelMode === "layers" ? "active" : ""} onClick={() => setPanelMode("layers")} role="tab">
              <Layers3 size={17} />Evidence
            </button>
          </div>

          {panelMode === "cities" ? (
            <section className="city-browser">
              <label className="search-box">
                <Search size={17} />
                <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search city or state" />
                {search && <button onClick={() => setSearch("")} title="Clear search" aria-label="Clear search"><X size={16} /></button>}
              </label>
              <div className="city-list-meta"><span>{visibleCities.length} cities</span><span>Canopy</span></div>
              <div className="city-list">
                {visibleCities.map((city) => (
                  <button className={selectedCity?.slug === city.slug ? "city-row selected" : "city-row"} key={city.slug} onClick={() => handleSelectCity(city)}>
                    <span className="rank">{String(city.rank).padStart(2, "0")}</span>
                    <span className="city-name"><strong>{city.name}</strong><small>{city.state}</small></span>
                    <span className="canopy-value">{city.areaWeightedCanopyPct.toFixed(1)}%</span>
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <section className="layer-panel">
              {selectedCity ? (
                <>
                  <div className="selected-heading">
                    <button className="icon-button subtle" onClick={handleNationalView} title="Return to the national city view" aria-label="Return to the national city view">
                      <ChevronLeft size={19} />
                    </button>
                    <div><h1>{selectedCity.name}</h1><p>{selectedCity.state} | population rank {selectedCity.rank}</p></div>
                  </div>

                  <div className="metric-grid">
                    <div><span>Population</span><strong>{formatPopulation(selectedCity.population)}</strong></div>
                    <div><span>Census tracts</span><strong>{formatPopulation(selectedCity.tracts)}</strong></div>
                    <div><span>Block groups</span><strong>{formatPopulation(selectedSummary?.neighbourhoods ?? selectedCity.blockGroups)}</strong></div>
                    <div><span>Canopy cover</span><strong>{selectedCity.areaWeightedCanopyPct.toFixed(1)}%</strong></div>
                  </div>

                  <div className="control-section evidence-control">
                    <h2>Neighbourhood evidence</h2>
                    <label className="select-row">
                      <span>Map view</span>
                      <select value={neighbourhoodOverlay} onChange={(event) => handleOverlay(event.target.value as NeighbourhoodOverlay)}>
                        {OVERLAYS.map((overlay) => <option key={overlay} value={overlay}>{OVERLAY_LABELS[overlay]}</option>)}
                      </select>
                    </label>

                    {neighbourhoodOverlay === "contexts" && selectedSummary && (
                      <>
                        <ContextSummary summary={selectedSummary} />
                        <div className="quick-filter" aria-label="Context filter">
                          <button className={contextFilter === "all" ? "active" : ""} onClick={() => handleContextFilter("all")}>All</button>
                          <button className={contextFilter === "A" ? "active" : ""} onClick={() => handleContextFilter("A")}>Joint</button>
                          <button className={contextFilter === "strict" ? "active" : ""} onClick={() => handleContextFilter("strict")}>G3H3V3</button>
                        </div>
                        <div className="context-legend">
                          {CONTEXT_IDS.map((id) => (
                            <button key={id} className={contextFilter === id ? "active" : ""} onClick={() => handleContextFilter(contextFilter === id ? "all" : id)}>
                              <i style={{ backgroundColor: CONTEXT_META[id].color }} />
                              <span><b>{id}</b>{CONTEXT_META[id].shortTitle}</span>
                              <small>{formatPopulation(selectedSummary.contexts[id].neighbourhoods)}</small>
                            </button>
                          ))}
                        </div>
                        <TypeExplorer filter={contextFilter} onChange={handleContextFilter} />
                      </>
                    )}

                    {["canopy_gap", "surface_heat", "social_constraint"].includes(neighbourhoodOverlay) && (
                      <DimensionLegend overlay={neighbourhoodOverlay} />
                    )}

                    {neighbourhoodOverlay === "allocation" && (
                      <div className="allocation-controls">
                        <label className="select-row">
                          <span>Ranking rule</span>
                          <select value={allocationRule} onChange={(event) => setAllocationRule(event.target.value as AllocationRule)}>
                            {ALLOCATION_RULES.map((rule) => <option key={rule} value={rule}>{ALLOCATION_LABELS[rule]}</option>)}
                          </select>
                        </label>
                        <div className="allocation-ramp"><i /><span>0</span><span>Share of local analytical capacity</span><span>100%</span></div>
                        <p>Every rule uses the same city-specific 10% area budget. Colour shows allocation to the block group, not a planting footprint.</p>
                      </div>
                    )}
                  </div>

                  <div className="control-section">
                    <h2>Map layers</h2>
                    <label className="toggle-row">
                      <input type="checkbox" checked={canopyVisible} onChange={(event) => setCanopyVisible(event.target.checked)} />
                      <span className="toggle-visual" /><span>0.6 m canopy mask</span>
                    </label>
                    <label className="toggle-row">
                      <input type="checkbox" checked={boundaryVisible} onChange={(event) => setBoundaryVisible(event.target.checked)} />
                      <span className="toggle-visual" /><span>City boundary</span>
                    </label>
                    <label className="range-row">
                      <span>Canopy opacity</span><output>{Math.round(canopyOpacity * 100)}%</output>
                      <input type="range" min="0.2" max="1" step="0.05" value={canopyOpacity} onChange={(event) => setCanopyOpacity(Number(event.target.value))} />
                    </label>
                    {neighbourhoodOverlay !== "none" && (
                      <label className="range-row">
                        <span>Neighbourhood opacity</span><output>{Math.round(neighbourhoodOpacity * 100)}%</output>
                        <input type="range" min="0.25" max="0.9" step="0.05" value={neighbourhoodOpacity} onChange={(event) => setNeighbourhoodOpacity(Number(event.target.value))} />
                      </label>
                    )}
                  </div>

                  <div className="inspection">
                    <h2><LocateFixed size={16} />{neighbourhoodOverlay === "none" ? "Canopy point query" : "Neighbourhood details"}</h2>
                    {neighbourhoodOverlay === "none" ? (
                      inspectionLoading ? <p>Reading this location...</p> : inspection ? (
                        <><strong>{classificationLabel(inspection.classification)}</strong><span>{inspection.longitude.toFixed(5)}, {inspection.latitude.toFixed(5)}</span></>
                      ) : <p>Click the map to inspect its canopy classification.</p>
                    ) : (
                      <NeighbourhoodInspection value={neighbourhoodInspection} allocationRule={allocationRule} />
                    )}
                  </div>

                  <div className="control-section metadata">
                    <h2>Imagery dates</h2>
                    <dl>
                      <div><dt>Median date</dt><dd>{selectedCity.imagery.median ?? "Pending"}</dd></div>
                      <div><dt>Date range</dt><dd>{selectedCity.imagery.earliest && selectedCity.imagery.latest ? `${selectedCity.imagery.earliest} to ${selectedCity.imagery.latest}` : "Pending"}</dd></div>
                    </dl>
                  </div>
                </>
              ) : (
                <div className="empty-panel">
                  <MapPinned size={28} /><h1>No city selected</h1><p>Choose a city to inspect canopy and neighbourhood evidence.</p>
                  <button className="primary-command" onClick={() => setPanelMode("cities")}>Browse 50 cities</button>
                </div>
              )}
            </section>
          )}
        </aside>

        <section className="map-region">
          <Suspense fallback={<div className="map-loading" role="status">Loading map...</div>}>
            <MapView
              cities={cities}
              selectedCity={selectedCity}
              canopyVisible={canopyVisible}
              boundaryVisible={boundaryVisible}
              canopyOpacity={canopyOpacity}
              neighbourhoodOverlay={neighbourhoodOverlay}
              neighbourhoodOpacity={neighbourhoodOpacity}
              contextFilter={contextFilter}
              allocationRule={allocationRule}
              selectedNeighbourhood={neighbourhoodInspection}
              onSelectCity={handleSelectCity}
              onInspect={handleInspect}
              onInspectNeighbourhood={handleInspectNeighbourhood}
            />
          </Suspense>
        </section>
      </div>

      {aboutOpen && (
        <div className="dialog-backdrop" role="presentation" onMouseDown={() => setAboutOpen(false)}>
          <section className="about-dialog" role="dialog" aria-modal="true" aria-labelledby="about-title" onMouseDown={(event) => event.stopPropagation()}>
            <button className="icon-button dialog-close" onClick={() => setAboutOpen(false)} title="Close" aria-label="Close"><X size={20} /></button>
            <h1 id="about-title">About this atlas</h1>
            <p>
              This atlas links the study's 0.6 m visible-woody-crown maps to the prespecified neighbourhood typology. The canopy layer is not cast shade. Surface LST is not personal heat exposure, and analytical allocation is not plantable land or a planting plan.
            </p>
            <p>
              For legibility, the web canopy layer suppresses 8-connected tree patches of 1–5 native pixels and masks pixels beneath basemap polygons classified as ocean or river. These display-only treatments do not alter the city or neighbourhood statistics reported in the manuscript.
            </p>
            <dl className="about-facts">
              <div><dt>Coverage</dt><dd>50 populous US cities</dd></div>
              <div><dt>Canopy atlas</dt><dd>38,618 municipal intersections</dd></div>
              <div><dt>Primary typology</dt><dd>34,566 fully contained block groups</dd></div>
              <div><dt>Classification</dt><dd>27 nominal G × H × V types</dd></div>
              <div><dt>Evidence release</dt><dd>{summaries?.freezeId ?? "GHV27-2026-08-09"}</dd></div>
            </dl>
            <p className="about-note">
              V uses poverty, renter occupancy, households without vehicles and age 65 years or older. Race and ethnicity do not enter V. The six contexts are descriptive combinations rather than an ordered risk index.
            </p>
          </section>
        </div>
      )}
    </main>
  );
}
