import type { AllocationRule, ContextId, NeighbourhoodOverlay } from "./types";

export const CONTEXT_META: Record<
  ContextId,
  { title: string; shortTitle: string; color: string; description: string }
> = {
  A: {
    title: "Joint canopy-deficit, surface-heat and social context",
    shortTitle: "Joint context",
    color: "#a83f3f",
    description: "Positive canopy deficit coincides with the upper surface-heat class and upper social-condition tier.",
  },
  B: {
    title: "Surface-heat canopy deficit",
    shortTitle: "Heat + canopy deficit",
    color: "#d47b2e",
    description: "Canopy deficit and the upper surface-heat class coincide without the upper social-condition tier.",
  },
  C: {
    title: "Social-context canopy deficit",
    shortTitle: "Social context + canopy deficit",
    color: "#74578d",
    description: "Canopy deficit and the upper social-condition tier coincide without the upper surface-heat class.",
  },
  D: {
    title: "Canopy-deficit dominant",
    shortTitle: "Canopy-deficit dominant",
    color: "#2b8075",
    description: "A positive canopy deficit occurs without either the upper surface-heat class or upper social-condition tier.",
  },
  E: {
    title: "Heat or social context with no measured canopy deficit",
    shortTitle: "Limited canopy deficit",
    color: "#547da6",
    description: "Surface heat or the social-condition ordering is high, but the measured canopy deficit is zero.",
  },
  F: {
    title: "Lower or mixed reference",
    shortTitle: "Lower or mixed",
    color: "#aeb8b3",
    description: "No measured canopy deficit and neither surface heat nor social conditions are in their upper class.",
  },
};

export const OVERLAY_LABELS: Record<NeighbourhoodOverlay, string> = {
  none: "Tree canopy",
  canopy_gap: "Canopy deficit",
  social_constraint: "Canopy and social conditions",
  surface_heat: "Canopy and surface heat",
  contexts: "Six neighbourhood contexts",
  allocation: "10% analytical budget",
};

export const ALLOCATION_LABELS: Record<AllocationRule, string> = {
  uniform: "Uniform fraction",
  canopy_gap: "Canopy-deficit first",
  surface_heat: "Surface-heat first",
  social_constraint: "Social-context first",
};

export const ALLOCATION_PROPERTIES: Record<AllocationRule, "a_u" | "a_g" | "a_h" | "a_v"> = {
  uniform: "a_u",
  canopy_gap: "a_g",
  surface_heat: "a_h",
  social_constraint: "a_v",
};

const G2_CONTEXT_COLORS: Partial<Record<ContextId, string>> = {
  A: "#c87570",
  B: "#e1a05d",
  C: "#9984aa",
  D: "#61a198",
};

export function typeCellColor(context: ContextId, gLevel: number) {
  if (gLevel === 2) return G2_CONTEXT_COLORS[context] ?? CONTEXT_META[context].color;
  return CONTEXT_META[context].color;
}

export function contextTitle(id: string | undefined) {
  return id && id in CONTEXT_META ? CONTEXT_META[id as ContextId].title : "Unclassified context";
}

export function levelLabel(dimension: "G" | "H" | "V", level: number) {
  if (dimension === "G") {
    return ["", "G1: no measured deficit", "G2: smaller positive deficit", "G3: larger positive deficit"][level];
  }
  if (dimension === "H") {
    return ["", "H1: no positive anomaly", "H2: smaller positive anomaly", "H3: larger positive anomaly"][level];
  }
  return ["", "V1: lower third", "V2: middle third", "V3: upper third"][level];
}
