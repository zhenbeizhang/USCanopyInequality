import { useEffect, useRef, useState } from "react";
import {
  AttributionControl,
  FullscreenControl,
  GeoJSONSource,
  Map as MapLibreMap,
  MapLayerMouseEvent,
  MapMouseEvent,
  NavigationControl,
  Popup,
  ScaleControl,
  setWorkerUrl,
} from "maplibre-gl";
import type {
  DataDrivenPropertyValueSpecification,
  ExpressionSpecification,
  FilterSpecification,
} from "@maplibre/maplibre-gl-style-spec";
import { ALLOCATION_PROPERTIES, CONTEXT_META, contextTitle } from "./evidence";
import type {
  AllocationRule,
  City,
  ContextFilter,
  NeighbourhoodOverlay,
  NeighbourhoodProperties,
  PointInspection,
} from "./types";

const BASEMAP_STYLE =
  import.meta.env.VITE_BASEMAP_STYLE ??
  (import.meta.env.PROD
    ? "/basemap/styles/positron?rev=2"
    : "https://tiles.openfreemap.org/styles/positron");
const TILE_API =
  import.meta.env.VITE_TILE_API ??
  (import.meta.env.PROD ? "" : "http://127.0.0.1:8000");
const COUNTRY_LABEL_LAYERS = new Set(["label_country_1", "label_country_2", "label_country_3"]);
const CANOPY_COLOR_STOPS: Array<[number, string]> = [
  [0, "#f1d58a"],
  [10, "#c9dca8"],
  [20, "#82bd9d"],
  [30, "#3d9479"],
  [45, "#0b5948"],
];

setWorkerUrl(`${import.meta.env.BASE_URL}vendor-v2/maplibre-gl-worker.mjs`);

interface MapViewProps {
  cities: City[];
  selectedCity: City | null;
  canopyVisible: boolean;
  boundaryVisible: boolean;
  canopyOpacity: number;
  neighbourhoodOverlay: NeighbourhoodOverlay;
  neighbourhoodOpacity: number;
  contextFilter: ContextFilter;
  allocationRule: AllocationRule;
  selectedNeighbourhood: NeighbourhoodProperties | null;
  onSelectCity: (city: City) => void;
  onInspect: (inspection: PointInspection | null, loading: boolean) => void;
  onInspectNeighbourhood: (inspection: NeighbourhoodProperties | null) => void;
}

function pointCollection(cities: City[]) {
  return {
    type: "FeatureCollection" as const,
    features: cities.map((city) => ({
      type: "Feature" as const,
      properties: {
        slug: city.slug,
        name: city.name,
        state: city.state,
        canopy: city.areaWeightedCanopyPct,
        available: city.raster.available,
      },
      geometry: {
        type: "Point" as const,
        coordinates: city.center,
      },
    })),
  };
}

type ScreenRect = {
  left: number;
  right: number;
  top: number;
  bottom: number;
};

function emptyFeatureCollection() {
  return {
    type: "FeatureCollection" as const,
    features: [],
  };
}

function rectanglesOverlap(first: ScreenRect, second: ScreenRect, padding = 0) {
  return !(
    first.right + padding <= second.left ||
    first.left >= second.right + padding ||
    first.bottom + padding <= second.top ||
    first.top >= second.bottom + padding
  );
}

function cityLabelFontSize(zoom: number) {
  if (zoom <= 3) return 9.5;
  if (zoom <= 6) return 9.5 + ((zoom - 3) / 3) * 3;
  if (zoom <= 10) return 12.5 + ((zoom - 6) / 4) * 2.5;
  return 15;
}

function cityPointRadius(zoom: number) {
  if (zoom <= 3) return 8;
  if (zoom >= 7) return 10;
  return 8 + ((zoom - 3) / 4) * 2;
}

function interpolateCanopyColor(value: number) {
  const bounded = Math.max(CANOPY_COLOR_STOPS[0][0], Math.min(45, value));
  const upperIndex = CANOPY_COLOR_STOPS.findIndex(([stop]) => stop >= bounded);
  if (upperIndex <= 0) return CANOPY_COLOR_STOPS[0][1];
  const [lowerStop, lowerColor] = CANOPY_COLOR_STOPS[upperIndex - 1];
  const [upperStop, upperColor] = CANOPY_COLOR_STOPS[upperIndex];
  const fraction = (bounded - lowerStop) / (upperStop - lowerStop);
  const parse = (color: string) => [
    Number.parseInt(color.slice(1, 3), 16),
    Number.parseInt(color.slice(3, 5), 16),
    Number.parseInt(color.slice(5, 7), 16),
  ];
  const lower = parse(lowerColor);
  const upper = parse(upperColor);
  const channel = (index: number) => Math.round(lower[index] + (upper[index] - lower[index]) * fraction);
  return `#${[channel(0), channel(1), channel(2)]
    .map((component) => component.toString(16).padStart(2, "0"))
    .join("")}`;
}

function cityCanopyColorExpression(cities: City[]): ExpressionSpecification {
  const expression: unknown[] = ["match", ["get", "slug"]];
  for (const city of cities) {
    expression.push(city.slug, interpolateCanopyColor(city.areaWeightedCanopyPct));
  }
  expression.push("#8aa49a");
  return expression as ExpressionSpecification;
}

function updateStudyCityLabelGeometry(
  map: MapLibreMap,
  cities: City[],
  selectedCity: City | null,
) {
  const labelSource = map.getSource("study-city-label-points") as GeoJSONSource | undefined;
  const leaderSource = map.getSource("study-city-label-leaders") as GeoJSONSource | undefined;
  if (!labelSource || !leaderSource) return;

  if (selectedCity) {
    labelSource.setData({
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: {
            slug: selectedCity.slug,
            name: selectedCity.name,
            state: selectedCity.state,
          },
          geometry: { type: "Point", coordinates: selectedCity.center },
        },
      ],
    });
    leaderSource.setData(emptyFeatureCollection());
    return;
  }

  const canvas = map.getCanvas();
  const mapWidth = canvas.clientWidth;
  const mapHeight = canvas.clientHeight;
  if (!mapWidth || !mapHeight) return;

  const zoom = map.getZoom();
  const fontSize = cityLabelFontSize(zoom);
  const pointRadius = cityPointRadius(zoom) + 2;
  const textCanvas = document.createElement("canvas");
  const context = textCanvas.getContext("2d");
  if (context) context.font = `400 ${fontSize}px Arial, sans-serif`;

  const projectedCities = cities.map((city) => ({
    city,
    point: map.project(city.center),
  }));
  const pointObstacles: ScreenRect[] = projectedCities.map(({ point }) => ({
    left: point.x - pointRadius,
    right: point.x + pointRadius,
    top: point.y - pointRadius,
    bottom: point.y + pointRadius,
  }));
  const densityBySlug = new Map(
    projectedCities.map(({ city, point }) => {
      const nearby = projectedCities.filter(({ city: otherCity, point: otherPoint }) => {
        if (otherCity.slug === city.slug) return false;
        return Math.hypot(point.x - otherPoint.x, point.y - otherPoint.y) < 85;
      }).length;
      return [city.slug, nearby];
    }),
  );

  const placedLabels: ScreenRect[] = [];
  const labelFeatures: Array<Record<string, unknown>> = [];
  const leaderFeatures: Array<Record<string, unknown>> = [];
  const orderedCities = [...projectedCities].sort((first, second) => {
    const densityDifference =
      (densityBySlug.get(second.city.slug) ?? 0) - (densityBySlug.get(first.city.slug) ?? 0);
    return densityDifference || first.city.rank - second.city.rank;
  });
  const margin = 6;

  for (const { city, point } of orderedCities) {
    const textWidth = Math.ceil(context?.measureText(city.name).width ?? city.name.length * 5.6) + 5;
    const textHeight = Math.ceil(fontSize + 5);
    const halfWidth = textWidth / 2;
    const halfHeight = textHeight / 2;
    const baseGap = pointRadius + 7;
    const candidates: Array<{ x: number; y: number; leader: boolean; cost: number }> = [];

    const addCandidate = (x: number, y: number, leader: boolean, cost: number) => {
      candidates.push({ x, y, leader, cost });
    };

    addCandidate(point.x + baseGap + halfWidth, point.y, false, 0);
    addCandidate(point.x - baseGap - halfWidth, point.y, false, 1);
    addCandidate(point.x, point.y - baseGap - halfHeight, true, 2);
    addCandidate(point.x, point.y + baseGap + halfHeight, true, 3);

    const rowStep = textHeight + 3;
    for (const extension of [8, 18, 30, 42]) {
      const horizontalDistance = baseGap + halfWidth + extension;
      const verticalDistance = baseGap + halfHeight + extension;
      for (const row of [0, -1, 1, -2, 2]) {
        const y = point.y + row * rowStep;
        addCandidate(point.x + horizontalDistance, y, true, extension + Math.abs(row) * 4);
        addCandidate(point.x - horizontalDistance, y, true, extension + Math.abs(row) * 4 + 1);
      }
      for (const column of [-1, 1, -2, 2]) {
        const x = point.x + column * Math.min(halfWidth + 7, 42);
        addCandidate(x, point.y - verticalDistance, true, extension + Math.abs(column) * 5 + 2);
        addCandidate(x, point.y + verticalDistance, true, extension + Math.abs(column) * 5 + 3);
      }
    }

    let bestCandidate: (typeof candidates)[number] | null = null;
    let bestPenalty = Number.POSITIVE_INFINITY;
    for (const candidate of candidates) {
      const rectangle: ScreenRect = {
        left: candidate.x - halfWidth,
        right: candidate.x + halfWidth,
        top: candidate.y - halfHeight,
        bottom: candidate.y + halfHeight,
      };
      const outsidePixels =
        Math.max(0, margin - rectangle.left) +
        Math.max(0, rectangle.right - (mapWidth - margin)) +
        Math.max(0, margin - rectangle.top) +
        Math.max(0, rectangle.bottom - (mapHeight - margin));
      const pointCollisions = pointObstacles.filter((obstacle) =>
        rectanglesOverlap(rectangle, obstacle, 1),
      ).length;
      const labelCollisions = placedLabels.filter((placed) =>
        rectanglesOverlap(rectangle, placed, 2),
      ).length;
      const penalty = outsidePixels * 1000 + pointCollisions * 10000 + labelCollisions * 20000 + candidate.cost;
      if (penalty < bestPenalty) {
        bestPenalty = penalty;
        bestCandidate = candidate;
      }
      if (!outsidePixels && !pointCollisions && !labelCollisions) break;
    }
    if (!bestCandidate) continue;

    const labelRectangle: ScreenRect = {
      left: bestCandidate.x - halfWidth,
      right: bestCandidate.x + halfWidth,
      top: bestCandidate.y - halfHeight,
      bottom: bestCandidate.y + halfHeight,
    };
    placedLabels.push(labelRectangle);
    const labelLngLat = map.unproject([bestCandidate.x, bestCandidate.y]);
    labelFeatures.push({
      type: "Feature",
      properties: { slug: city.slug, name: city.name, state: city.state },
      geometry: { type: "Point", coordinates: [labelLngLat.lng, labelLngLat.lat] },
    });

    if (bestCandidate.leader) {
      const dx = bestCandidate.x - point.x;
      const dy = bestCandidate.y - point.y;
      const distance = Math.max(Math.hypot(dx, dy), 1);
      const unitX = dx / distance;
      const unitY = dy / distance;
      const start = [point.x + unitX * pointRadius, point.y + unitY * pointRadius] as [number, number];
      const edgeDistance = Math.min(
        Math.abs(unitX) > 0.001 ? halfWidth / Math.abs(unitX) : Number.POSITIVE_INFINITY,
        Math.abs(unitY) > 0.001 ? halfHeight / Math.abs(unitY) : Number.POSITIVE_INFINITY,
      );
      const end = [
        bestCandidate.x - unitX * edgeDistance,
        bestCandidate.y - unitY * edgeDistance,
      ] as [number, number];
      const startLngLat = map.unproject(start);
      const endLngLat = map.unproject(end);
      leaderFeatures.push({
        type: "Feature",
        properties: { slug: city.slug },
        geometry: {
          type: "LineString",
          coordinates: [
            [startLngLat.lng, startLngLat.lat],
            [endLngLat.lng, endLngLat.lat],
          ],
        },
      });
    }
  }

  labelSource.setData({ type: "FeatureCollection", features: labelFeatures });
  leaderSource.setData({ type: "FeatureCollection", features: leaderFeatures });
}

function retainCountryLabelsOnly(map: MapLibreMap) {
  for (const layer of map.getStyle().layers ?? []) {
    if (layer.type === "symbol" && !COUNTRY_LABEL_LAYERS.has(layer.id)) {
      map.setLayoutProperty(layer.id, "visibility", "none");
    }
  }
}

function nationalBounds(cities: City[]): [[number, number], [number, number]] | null {
  if (!cities.length) return null;
  return [
    [
      Math.min(...cities.map((city) => city.bounds[0])),
      Math.min(...cities.map((city) => city.bounds[1])),
    ],
    [
      Math.max(...cities.map((city) => city.bounds[2])),
      Math.max(...cities.map((city) => city.bounds[3])),
    ],
  ];
}

function showNationalView(map: MapLibreMap, cities: City[], duration = 750) {
  const bounds = nationalBounds(cities);
  if (!bounds) {
    map.easeTo({ center: [-98.5, 38.4], zoom: 3.2, duration });
    return;
  }
  map.fitBounds(bounds, {
    padding: { top: 44, right: 44, bottom: 44, left: 44 },
    duration,
    maxZoom: 3.75,
  });
}

function contextColorExpression(): ExpressionSpecification {
  return [
    "match",
    ["get", "context"],
    "A",
    CONTEXT_META.A.color,
    "B",
    CONTEXT_META.B.color,
    "C",
    CONTEXT_META.C.color,
    "D",
    CONTEXT_META.D.color,
    "E",
    CONTEXT_META.E.color,
    "F",
    CONTEXT_META.F.color,
    "#aeb8b3",
  ] as ExpressionSpecification;
}

function levelColorExpression(property: "gl" | "hl" | "vl"): ExpressionSpecification {
  const palettes = {
    gl: ["#e7ece9", "#e2ae61", "#aa4b3e"],
    hl: ["#e7ece9", "#e29a55", "#b43e35"],
    vl: ["#e7edf1", "#83a8bc", "#3f6d8a"],
  };
  const colors = palettes[property];
  return [
    "match",
    ["get", property],
    1,
    colors[0],
    2,
    colors[1],
    3,
    colors[2],
    "#d9dfdc",
  ] as ExpressionSpecification;
}

function contextFilterExpression(filter: ContextFilter): FilterSpecification | null {
  if (filter === "all") return null;
  if (filter === "strict") {
    return ["==", ["get", "type"], "G3H3V3"] as FilterSpecification;
  }
  if (filter.startsWith("G")) {
    return ["==", ["get", "type"], filter] as FilterSpecification;
  }
  return ["==", ["get", "context"], filter] as FilterSpecification;
}

function normaliseProperties(properties: Record<string, unknown>): NeighbourhoodProperties {
  return {
    geoid: String(properties.geoid),
    pop: Number(properties.pop),
    canopy: Number(properties.canopy),
    g: Number(properties.g),
    h: Number(properties.h),
    v: Number(properties.v),
    gl: Number(properties.gl) as 1 | 2 | 3,
    hl: Number(properties.hl) as 1 | 2 | 3,
    vl: Number(properties.vl) as 1 | 2 | 3,
    type: String(properties.type) as NeighbourhoodProperties["type"],
    context: String(properties.context) as NeighbourhoodProperties["context"],
    a_u: Number(properties.a_u),
    a_g: Number(properties.a_g),
    a_h: Number(properties.a_h),
    a_v: Number(properties.a_v),
  };
}

function updateNeighbourhoodStyle(
  map: MapLibreMap,
  overlay: NeighbourhoodOverlay,
  opacity: number,
  filter: ContextFilter,
  allocationRule: AllocationRule,
) {
  if (!map.getLayer("neighbourhood-fill")) return;
  const visible = overlay === "none" ? "none" : "visible";
  for (const layer of ["neighbourhood-hit", "neighbourhood-fill", "neighbourhood-outline", "neighbourhood-selected"]) {
    if (map.getLayer(layer)) map.setLayoutProperty(layer, "visibility", visible);
  }

  let color: DataDrivenPropertyValueSpecification<string> = contextColorExpression();
  let fillOpacity: DataDrivenPropertyValueSpecification<number> = opacity;
  if (overlay === "canopy_gap") color = levelColorExpression("gl");
  if (overlay === "surface_heat") color = levelColorExpression("hl");
  if (overlay === "social_constraint") color = levelColorExpression("vl");
  if (overlay === "allocation") {
    const property = ALLOCATION_PROPERTIES[allocationRule];
    color = [
      "interpolate",
      ["linear"],
      ["get", property],
      0,
      "#dfe4e1",
      0.1,
      "#e4b85f",
      0.5,
      "#c86b37",
      1,
      "#813c3b",
    ] as ExpressionSpecification;
    fillOpacity = [
      "case",
      [">", ["get", property], 0],
      opacity,
      Math.min(0.12, opacity),
    ] as ExpressionSpecification;
  }
  map.setPaintProperty("neighbourhood-fill", "fill-color", color);
  map.setPaintProperty("neighbourhood-fill", "fill-opacity", fillOpacity);
  const activeFilter = overlay === "contexts" ? contextFilterExpression(filter) : null;
  for (const layer of ["neighbourhood-hit", "neighbourhood-fill", "neighbourhood-outline"]) {
    if (map.getLayer(layer)) map.setFilter(layer, activeFilter);
  }
}

export default function MapView({
  cities,
  selectedCity,
  canopyVisible,
  boundaryVisible,
  canopyOpacity,
  neighbourhoodOverlay,
  neighbourhoodOpacity,
  contextFilter,
  allocationRule,
  selectedNeighbourhood,
  onSelectCity,
  onInspect,
  onInspectNeighbourhood,
}: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const neighbourhoodPopupRef = useRef<Popup | null>(null);
  const [mapReady, setMapReady] = useState(false);
  const neighbourhoodsRequested = neighbourhoodOverlay !== "none";
  const selectedRef = useRef<City | null>(selectedCity);
  const citiesRef = useRef(cities);
  const overlayRef = useRef(neighbourhoodOverlay);

  useEffect(() => {
    selectedRef.current = selectedCity;
  }, [selectedCity]);

  useEffect(() => {
    citiesRef.current = cities;
  }, [cities]);

  useEffect(() => {
    overlayRef.current = neighbourhoodOverlay;
  }, [neighbourhoodOverlay]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new MapLibreMap({
      container: containerRef.current,
      style: BASEMAP_STYLE,
      center: [-98.5, 38.4],
      zoom: 3.35,
      minZoom: 2.5,
      maxZoom: 20,
      attributionControl: false,
    });
    const cityPopup = new Popup({ closeButton: false, closeOnClick: false, offset: 10 });
    const neighbourhoodPopup = new Popup({ closeButton: false, closeOnClick: false, offset: 8 });
    neighbourhoodPopupRef.current = neighbourhoodPopup;
    mapRef.current = map;
    map.addControl(new NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new FullscreenControl(), "top-right");
    map.addControl(new ScaleControl({ maxWidth: 110, unit: "metric" }), "bottom-right");
    map.addControl(
      new AttributionControl({
        compact: true,
        customAttribution: "Canopy and neighbourhood evidence © study authors",
      }),
      "bottom-right",
    );

    map.on("style.load", () => {
      if (mapRef.current !== map) return;
      retainCountryLabelsOnly(map);
      if (map.getLayer("background")) {
        map.setPaintProperty("background", "background-color", "#f2f3f1");
      }
      if (map.getLayer("water")) {
        map.setPaintProperty("water", "fill-color", "#dde6e8");
      }
      if (map.getLayer("waterway")) {
        map.setPaintProperty("waterway", "line-color", "#c4d3d7");
      }
      map.addSource("us-boundary", {
        type: "geojson",
        data: `${import.meta.env.BASE_URL}catalog/us_nation_20m.geojson`,
      });
      map.addLayer(
        {
          id: "us-focus-fill",
          type: "fill",
          source: "us-boundary",
          paint: {
            "fill-color": "#edf4f0",
            "fill-opacity": 0.9,
          },
        },
        map.getLayer("park") ? "park" : undefined,
      );
      map.addLayer({
        id: "us-boundary-casing",
        type: "line",
        source: "us-boundary",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": "#ffffff",
          "line-width": ["interpolate", ["linear"], ["zoom"], 3, 3.8, 7, 5.4],
          "line-opacity": 0.9,
        },
      });
      map.addLayer({
        id: "us-boundary-outline",
        type: "line",
        source: "us-boundary",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": "#26352f",
          "line-width": ["interpolate", ["linear"], ["zoom"], 3, 1.9, 7, 2.8],
          "line-opacity": 0.92,
        },
      });
      map.addSource("city-boundaries", {
        type: "geojson",
        data: `${import.meta.env.BASE_URL}catalog/city_boundaries.geojson`,
      });
      map.addLayer({
        id: "city-area-all",
        type: "fill",
        source: "city-boundaries",
        paint: {
          "fill-color": cityCanopyColorExpression(citiesRef.current),
          "fill-opacity": 0.24,
        },
      });
      map.addLayer({
        id: "city-boundary-all",
        type: "line",
        source: "city-boundaries",
        paint: {
          "line-color": "#0d654d",
          "line-width": ["interpolate", ["linear"], ["zoom"], 3, 1.05, 10, 1.45],
          "line-opacity": 0.82,
        },
      });
      map.addLayer({
        id: "city-boundary-selected",
        type: "line",
        source: "city-boundaries",
        filter: ["==", ["get", "slug"], ""],
        paint: {
          "line-color": "#111714",
          "line-width": ["interpolate", ["linear"], ["zoom"], 4, 1.5, 15, 2.4],
          "line-opacity": 0.95,
        },
      });
      map.addSource("city-points", {
        type: "geojson",
        data: pointCollection(citiesRef.current),
      });
      map.addSource("study-city-label-points", {
        type: "geojson",
        data: emptyFeatureCollection(),
      });
      map.addSource("study-city-label-leaders", {
        type: "geojson",
        data: emptyFeatureCollection(),
      });
      map.addLayer({
        id: "study-city-label-leaders",
        type: "line",
        source: "study-city-label-leaders",
        minzoom: 2.5,
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": "#596861",
          "line-width": 1.15,
          "line-opacity": 0.88,
        },
      });
      map.addLayer({
        id: "city-points-halo",
        type: "circle",
        source: "city-points",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 3, 8, 7, 10],
          "circle-color": "#ffffff",
          "circle-stroke-color": "#174f3f",
          "circle-stroke-width": 1.2,
          "circle-opacity": 0.98,
        },
      });
      map.addLayer({
        id: "city-points",
        type: "circle",
        source: "city-points",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 3, 5.1, 7, 6.7],
          "circle-color": [
            "case",
            ["boolean", ["get", "available"], false],
            [
              "interpolate",
              ["linear"],
              ["get", "canopy"],
              0,
              "#f1d58a",
              10,
              "#c9dca8",
              20,
              "#82bd9d",
              30,
              "#3d9479",
              45,
              "#0b5948",
            ],
            "#7d8984",
          ],
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 1.1,
        },
      });
      map.addLayer({
        id: "study-city-labels",
        type: "symbol",
        source: "study-city-label-points",
        minzoom: 2.5,
        layout: {
          "text-field": ["get", "name"],
          "text-font": ["Noto Sans Regular"],
          "text-size": ["interpolate", ["linear"], ["zoom"], 3, 9.5, 6, 12.5, 10, 15],
          "text-anchor": "center",
          "text-justify": "center",
          "text-max-width": 30,
          "text-padding": 0,
          "text-allow-overlap": true,
          "text-ignore-placement": true,
        },
        paint: {
          "text-color": "#173d32",
          "text-halo-color": "#ffffff",
          "text-halo-width": 1.4,
          "text-halo-blur": 0.4,
        },
      });
      const refreshStudyCityLabels = () =>
        updateStudyCityLabelGeometry(map, citiesRef.current, selectedRef.current);
      map.on("moveend", refreshStudyCityLabels);
      map.on("resize", refreshStudyCityLabels);
      refreshStudyCityLabels();

      map.on("mouseenter", "city-points", (event: MapLayerMouseEvent) => {
        map.getCanvas().style.cursor = "pointer";
        const feature = event.features?.[0];
        if (!feature || feature.geometry.type !== "Point") return;
        const name = String(feature.properties?.name ?? "City");
        const state = String(feature.properties?.state ?? "");
        const canopy = Number(feature.properties?.canopy);
        const label = Number.isFinite(canopy)
          ? `${name}, ${state} | ${canopy.toFixed(1)}% canopy`
          : `${name}, ${state}`;
        cityPopup.setLngLat(feature.geometry.coordinates as [number, number]).setText(label).addTo(map);
      });
      map.on("mouseleave", "city-points", () => {
        map.getCanvas().style.cursor = "";
        cityPopup.remove();
      });
      map.on("click", "city-points", (event: MapLayerMouseEvent) => {
        const slug = event.features?.[0]?.properties?.slug as string | undefined;
        const city = citiesRef.current.find((item) => item.slug === slug);
        if (city) onSelectCity(city);
      });
      setMapReady(true);
    });

    map.on("mousemove", (event: MapMouseEvent) => {
      if (overlayRef.current === "none" || !map.getLayer("neighbourhood-hit")) {
        neighbourhoodPopup.remove();
        return;
      }
      const features = map.queryRenderedFeatures(event.point, { layers: ["neighbourhood-hit"] });
      const properties = features[0]?.properties;
      if (!properties) {
        neighbourhoodPopup.remove();
        map.getCanvas().style.cursor = "";
        return;
      }
      map.getCanvas().style.cursor = "pointer";
      neighbourhoodPopup
        .setLngLat(event.lngLat)
        .setText(`${String(properties.type)} | ${contextTitle(String(properties.context))}`)
        .addTo(map);
    });

    map.on("click", async (event: MapMouseEvent) => {
      const city = selectedRef.current;
      if (!city?.raster.available) return;
      const pointFeatures = map.queryRenderedFeatures(event.point, { layers: ["city-points"] });
      if (pointFeatures.length) return;

      const waterMaskLayers = ["canopy-water-mask", "canopy-river-line-mask"].filter((layer) =>
        Boolean(map.getLayer(layer)),
      );
      const maskedWaterFeatures = waterMaskLayers.length
        ? map.queryRenderedFeatures(event.point, { layers: waterMaskLayers })
        : [];
      if (maskedWaterFeatures.length) {
        onInspectNeighbourhood(null);
        if (map.getLayer("neighbourhood-selected")) {
          map.setFilter("neighbourhood-selected", ["==", ["get", "geoid"], ""]);
        }
        onInspect(
          {
            city: city.slug,
            longitude: event.lngLat.lng,
            latitude: event.lngLat.lat,
            value: 254,
            classification: "nodata",
          },
          false,
        );
        return;
      }

      if (overlayRef.current !== "none" && map.getLayer("neighbourhood-hit")) {
        const neighbourhoods = map.queryRenderedFeatures(event.point, {
          layers: ["neighbourhood-hit"],
        });
        const properties = neighbourhoods[0]?.properties;
        if (properties) {
          const inspection = normaliseProperties(properties);
          map.setFilter("neighbourhood-selected", ["==", ["get", "geoid"], inspection.geoid]);
          onInspect(null, false);
          onInspectNeighbourhood(inspection);
          return;
        }
      }

      onInspectNeighbourhood(null);
      onInspect(null, true);
      try {
        const response = await fetch(
          `${TILE_API}/api/value/${city.slug}?lon=${event.lngLat.lng}&lat=${event.lngLat.lat}`,
        );
        if (!response.ok) throw new Error("No value");
        onInspect((await response.json()) as PointInspection, false);
      } catch {
        onInspect(null, false);
      }
    });

    return () => {
      cityPopup.remove();
      neighbourhoodPopup.remove();
      neighbourhoodPopupRef.current = null;
      map.remove();
      mapRef.current = null;
      setMapReady(false);
    };
  }, [onInspect, onInspectNeighbourhood, onSelectCity]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!map.getLayer("city-boundary-selected")) return;

    const setVisibility = (layerId: string, visibility: "visible" | "none") => {
      if (map.getLayer(layerId)) map.setLayoutProperty(layerId, "visibility", visibility);
    };

    for (const layer of [
      "neighbourhood-selected",
      "neighbourhood-outline",
      "neighbourhood-fill",
      "neighbourhood-hit",
      "canopy-river-line-mask",
      "canopy-water-mask",
      "canopy",
    ]) {
      if (map.getLayer(layer)) map.removeLayer(layer);
    }
    for (const source of ["neighbourhoods", "canopy"]) {
      if (map.getSource(source)) map.removeSource(source);
    }

    if (selectedCity) {
      map.setFilter("city-boundary-selected", ["==", ["get", "slug"], selectedCity.slug]);
      setVisibility("us-focus-fill", "none");
      setVisibility("us-boundary-casing", "none");
      setVisibility("us-boundary-outline", "none");
      setVisibility("city-area-all", "none");
      setVisibility("city-points-halo", "none");
      setVisibility("city-points", "none");
      setVisibility("study-city-label-leaders", "none");
      updateStudyCityLabelGeometry(map, citiesRef.current, selectedCity);
      if (selectedCity.raster.available) {
        map.addSource("canopy", {
          type: "raster",
          tiles: [`${TILE_API}/tiles/${selectedCity.slug}/{z}/{x}/{y}.png`],
          tileSize: 256,
          minzoom: 7,
          maxzoom: 20,
          bounds: selectedCity.bounds,
        });
        map.addLayer(
          {
            id: "canopy",
            type: "raster",
            source: "canopy",
            layout: { visibility: canopyVisible ? "visible" : "none" },
            paint: {
              "raster-opacity": canopyOpacity,
              "raster-resampling": "nearest",
              "raster-fade-duration": 80,
            },
          },
          "city-boundary-selected",
        );
        if (map.getSource("openmaptiles")) {
          map.addLayer(
            {
              id: "canopy-water-mask",
              type: "fill",
              source: "openmaptiles",
              "source-layer": "water",
              filter: ["match", ["get", "class"], ["ocean", "river"], true, false],
              paint: {
                "fill-color": "#dde6e8",
                "fill-opacity": 1,
              },
            },
            "city-boundary-selected",
          );
          map.addLayer(
            {
              id: "canopy-river-line-mask",
              type: "line",
              source: "openmaptiles",
              "source-layer": "waterway",
              filter: ["==", ["get", "class"], "river"],
              paint: {
                "line-color": "#dde6e8",
                "line-width": ["interpolate", ["linear"], ["zoom"], 8, 1, 12, 1.5, 16, 3, 20, 5],
                "line-opacity": 1,
              },
            },
            "city-boundary-selected",
          );
        }
      }
      if (neighbourhoodsRequested) {
        map.addSource("neighbourhoods", {
          type: "geojson",
          data: `${import.meta.env.BASE_URL}neighbourhoods/${selectedCity.slug}.geojson`,
          promoteId: "geoid",
        });
        map.addLayer(
          {
            id: "neighbourhood-hit",
            type: "fill",
            source: "neighbourhoods",
            paint: { "fill-color": "#000000", "fill-opacity": 0 },
          },
          "city-boundary-selected",
        );
        map.addLayer(
          {
            id: "neighbourhood-fill",
            type: "fill",
            source: "neighbourhoods",
            paint: { "fill-color": contextColorExpression(), "fill-opacity": neighbourhoodOpacity },
          },
          "city-boundary-selected",
        );
        map.addLayer(
          {
            id: "neighbourhood-outline",
            type: "line",
            source: "neighbourhoods",
            paint: {
              "line-color": "#34443d",
              "line-width": ["interpolate", ["linear"], ["zoom"], 7, 0.25, 13, 0.7],
              "line-opacity": 0.34,
            },
          },
          "city-boundary-selected",
        );
        map.addLayer(
          {
            id: "neighbourhood-selected",
            type: "line",
            source: "neighbourhoods",
            filter: ["==", ["get", "geoid"], ""],
            paint: { "line-color": "#101513", "line-width": 2.4, "line-opacity": 1 },
          },
          "city-boundary-selected",
        );
        updateNeighbourhoodStyle(
          map,
          neighbourhoodOverlay,
          neighbourhoodOpacity,
          contextFilter,
          allocationRule,
        );
      }
      map.fitBounds(
        [
          [selectedCity.bounds[0], selectedCity.bounds[1]],
          [selectedCity.bounds[2], selectedCity.bounds[3]],
        ],
        { padding: 40, duration: 0, maxZoom: 12.5 },
      );
    } else {
      map.setFilter("city-boundary-selected", ["==", ["get", "slug"], ""]);
      setVisibility("us-focus-fill", "visible");
      setVisibility("us-boundary-casing", "visible");
      setVisibility("us-boundary-outline", "visible");
      setVisibility("city-area-all", "visible");
      setVisibility("city-points-halo", "visible");
      setVisibility("city-points", "visible");
      setVisibility("study-city-label-leaders", "visible");
      updateStudyCityLabelGeometry(map, citiesRef.current, null);
      showNationalView(map, citiesRef.current);
    }
  }, [selectedCity, mapReady, neighbourhoodsRequested]);

  useEffect(() => {
    const source = mapRef.current?.getSource("city-points") as GeoJSONSource | undefined;
    if (mapReady && source) {
      source.setData(pointCollection(cities));
      if (mapRef.current?.getLayer("city-area-all")) {
        mapRef.current.setPaintProperty(
          "city-area-all",
          "fill-color",
          cityCanopyColorExpression(cities),
        );
      }
      if (!selectedRef.current) showNationalView(mapRef.current!, cities, 0);
      updateStudyCityLabelGeometry(mapRef.current!, cities, selectedRef.current);
    }
  }, [cities, mapReady]);

  useEffect(() => {
    const map = mapRef.current;
    if (map?.getLayer("canopy")) {
      map.setLayoutProperty("canopy", "visibility", canopyVisible ? "visible" : "none");
      map.setPaintProperty("canopy", "raster-opacity", canopyOpacity);
    }
  }, [canopyVisible, canopyOpacity, mapReady]);

  useEffect(() => {
    const map = mapRef.current;
    if (map?.getLayer("city-boundary-all")) {
      map.setLayoutProperty("city-boundary-all", "visibility", boundaryVisible ? "visible" : "none");
      map.setLayoutProperty(
        "city-boundary-selected",
        "visibility",
        boundaryVisible ? "visible" : "none",
      );
    }
  }, [boundaryVisible, mapReady]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (neighbourhoodOverlay === "none") {
      neighbourhoodPopupRef.current?.remove();
      map.getCanvas().style.cursor = "";
    }
    updateNeighbourhoodStyle(
      map,
      neighbourhoodOverlay,
      neighbourhoodOpacity,
      contextFilter,
      allocationRule,
    );
  }, [neighbourhoodOverlay, neighbourhoodOpacity, contextFilter, allocationRule, mapReady]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.getLayer("neighbourhood-selected")) return;
    map.setFilter(
      "neighbourhood-selected",
      ["==", ["get", "geoid"], selectedNeighbourhood?.geoid ?? ""],
    );
  }, [selectedNeighbourhood, mapReady]);

  const canopyValues = cities.map((city) => city.areaWeightedCanopyPct).filter(Number.isFinite);
  const minimumCanopy = canopyValues.length ? Math.min(...canopyValues) : 0;
  const maximumCanopy = canopyValues.length ? Math.max(...canopyValues) : 45;

  return (
    <>
      <div
        className="map-canvas"
        ref={containerRef}
        aria-label="Interactive canopy and neighbourhood evidence map"
      />
      {!selectedCity && (
        <div className="canopy-rate-legend" aria-label="City canopy-cover colour scale">
          <strong>City canopy cover</strong>
          <span className="canopy-rate-ramp" aria-hidden />
          <div>
            <span>{minimumCanopy.toFixed(1)}%</span>
            <span>{maximumCanopy.toFixed(1)}%</span>
          </div>
        </div>
      )}
    </>
  );
}
