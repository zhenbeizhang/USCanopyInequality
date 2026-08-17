export type Bounds = [number, number, number, number];

export type ContextId = "A" | "B" | "C" | "D" | "E" | "F";
export type NeighbourhoodOverlay =
  | "none"
  | "contexts"
  | "canopy_gap"
  | "surface_heat"
  | "social_constraint"
  | "allocation";
export type AllocationRule = "uniform" | "canopy_gap" | "surface_heat" | "social_constraint";
export type ContextFilter = "all" | ContextId | "strict" | `G${1 | 2 | 3}H${1 | 2 | 3}V${1 | 2 | 3}`;

export interface City {
  rank: number;
  slug: string;
  cityKey: string;
  name: string;
  state: string;
  bounds: Bounds;
  center: [number, number];
  areaWeightedCanopyPct: number;
  populationWeightedCanopyPct: number;
  population: number;
  tracts: number;
  blockGroups: number;
  imagery: {
    earliest: string | null;
    median: string | null;
    latest: string | null;
    uniqueDates: number | null;
  };
  raster: {
    file: string;
    width?: number;
    height?: number;
    crs?: string;
    overviews?: number[];
    available: boolean;
  };
}

export interface PointInspection {
  city: string;
  longitude: number;
  latitude: number;
  value: number;
  classification: "canopy" | "not_canopy" | "nodata";
}

export interface NeighbourhoodProperties {
  geoid: string;
  pop: number;
  canopy: number;
  g: number;
  h: number;
  v: number;
  gl: 1 | 2 | 3;
  hl: 1 | 2 | 3;
  vl: 1 | 2 | 3;
  type: `G${1 | 2 | 3}H${1 | 2 | 3}V${1 | 2 | 3}`;
  context: ContextId;
  a_u: number;
  a_g: number;
  a_h: number;
  a_v: number;
}

export interface NeighbourhoodSummary {
  slug: string;
  name: string;
  neighbourhoods: number;
  population: number;
  jointContext: {
    neighbourhoods: number;
    population: number;
    observedExpectedRatio: number | null;
  };
  strictTripleHigh: {
    neighbourhoods: number;
    population: number;
    observedExpectedRatio: number | null;
  };
  contexts: Record<ContextId, { neighbourhoods: number; population: number }>;
}

export interface NeighbourhoodSummaryCatalog {
  freezeId: string;
  spatialUnit: string;
  totalNeighbourhoods: number;
  primaryBudgetPct: number;
  cities: Record<string, NeighbourhoodSummary>;
}
