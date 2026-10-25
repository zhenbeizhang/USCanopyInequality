from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import ipaddress
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ATLAS_URL = "https://atlas.8-217-164-241.sslip.io/atlas/"
SUPPORTING_DATA_DOI = "https://doi.org/10.5281/zenodo.21976809"
CURRENT_RELEASE_VERSION = "0.3.1"
PUBLIC_ATLAS_DOCS = {
    Path("README.md"),
    Path("atlas/README.md"),
    Path("pyproject.toml"),
}
DOCUMENTED_PUBLIC_IPV4 = {
    "223.5.5.5": {Path("atlas/deploy/who-has-shade.nginx")},
    "1.1.1.1": {Path("atlas/deploy/who-has-shade.nginx")},
}
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "runtime",
    "work",
    "__pycache__",
}


def project_files(pattern: str):
    return (
        path
        for path in ROOT.rglob(pattern)
        if not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


class StaticReleaseTests(unittest.TestCase):
    def test_python_sources_compile(self) -> None:
        for path in project_files("*.py"):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_no_workstation_absolute_path_literals(self) -> None:
        failures = []
        for path in project_files("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if re.match(r"^[A-Za-z]:[\\/]", node.value):
                        failures.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual([], failures)

    def test_no_personal_workspace_markers(self) -> None:
        forbidden = ("whu" + "zzb", "美国" + "树荫")
        text_suffixes = {
            ".cff",
            ".csv",
            ".example",
            ".json",
            ".md",
            ".nginx",
            ".ps1",
            ".py",
            ".service",
            ".toml",
            ".ts",
            ".tsx",
            ".txt",
            ".yaml",
            ".yml",
        }
        failures = []
        for path in project_files("*"):
            if not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            for token in forbidden:
                if token.lower() in text:
                    failures.append(f"{path.relative_to(ROOT)}: {token}")
        self.assertEqual([], failures)

    def test_no_private_binary_artifacts(self) -> None:
        forbidden = {".pth", ".pt", ".onnx", ".tif", ".tiff", ".gpkg", ".docx"}
        files = [
            str(path.relative_to(ROOT))
            for path in project_files("*")
            if path.is_file() and path.suffix.lower() in forbidden
        ]
        self.assertEqual([], files)

    def test_expected_figure_scripts(self) -> None:
        main = list((ROOT / "scripts" / "06_main_figures").glob("make_figure_[1-5].py"))
        supplement = list((ROOT / "scripts" / "07_supplementary_figures").glob("make_supplementary_figure_*.py"))
        self.assertEqual(5, len(main))
        self.assertEqual(16, len(supplement))

    def test_pipeline_inventory_is_complete(self) -> None:
        spec = importlib.util.spec_from_file_location("release_pipeline", ROOT / "run_pipeline.py")
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        scripts = [name for names in module.STAGES.values() for name in names]
        self.assertEqual(81, len(scripts))
        self.assertEqual(81, len(set(scripts)))
        self.assertTrue(all((ROOT / "scripts" / name).is_file() for name in scripts))

    def test_protocols_and_targets_are_declared(self) -> None:
        for name in (
            "TRADEOFF_OUTCOMES_PROTOCOL.md",
            "DIVERGENCE_DECOMPOSITION_PROTOCOL.md",
            "DIVERGENCE_IMPLEMENTATION_NOTE.md",
            "CROSS_CITY_TRANSFERABILITY_PROTOCOL.md",
        ):
            self.assertTrue((ROOT / "docs" / "protocols" / name).is_file(), name)

        with (ROOT / "manifests" / "analysis_targets.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            targets = list(csv.DictReader(stream))
        quantities = {row["quantity"] for row in targets}
        for quantity in (
            "ten-percent heat-social allocation overlap",
            "ten-percent redirected area translation",
            "ten-percent heat objective forgone under social rule",
            "ten-percent social objective forgone under heat rule",
            "ten-percent cross-city transferability RMSE ratio",
        ):
            self.assertIn(quantity, quantities)

    def test_public_release_has_no_deployment_secrets(self) -> None:
        forbidden = (
            "who-has-shade-deploy.pem",
            "root@",
        )
        ipv4_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        failures = []
        for suffix in ("*.md", "*.ps1", "*.service", "*.nginx", "*.json", "*.toml"):
            for path in project_files(suffix):
                text = path.read_text(encoding="utf-8", errors="replace")
                relative = path.relative_to(ROOT)
                for token in forbidden:
                    if token in text:
                        failures.append(f"{relative}: {token}")
                for address in ipv4_pattern.findall(text):
                    if not ipaddress.ip_address(address).is_global:
                        continue
                    if relative in DOCUMENTED_PUBLIC_IPV4.get(address, set()):
                        continue
                    failures.append(f"{relative}: undocumented public IPv4 address {address}")
        self.assertEqual([], failures)

    def test_public_atlas_entry_is_documented(self) -> None:
        for relative in PUBLIC_ATLAS_DOCS:
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(PUBLIC_ATLAS_URL, text, str(relative))

    def test_published_data_release_documents_are_current(self) -> None:
        release_docs = (
            Path("README.md"),
            Path("data/README.md"),
            Path("docs/code_availability_statement.md"),
            Path("docs/data_contract.md"),
            Path("docs/release_plan.md"),
            Path("docs/reproducibility_scope.md"),
        )
        stale_phrases = (
            "# Data placeholder",
            "will be released as a separate",
            "Before archival DOI assignment",
            "Replace the bracketed fields",
            "No DOI or accession is asserted",
            "Deferred to the separate data release",
        )
        for relative in release_docs:
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(SUPPORTING_DATA_DOI, text, str(relative))
            for phrase in stale_phrases:
                self.assertNotIn(phrase, text, f"{relative}: {phrase}")

        for relative in (
            Path("docs/workflow.md"),
            Path("docs/reproducibility_scope.md"),
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("seventeen Supplementary figures", text, str(relative))

    def test_release_version_is_consistent(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        atlas_package = json.loads(
            (ROOT / "atlas" / "package.json").read_text(encoding="utf-8")
        )
        audit = json.loads(
            (ROOT / "manifests" / "code_release_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn(f'version = "{CURRENT_RELEASE_VERSION}"', pyproject)
        self.assertIn(f"version: {CURRENT_RELEASE_VERSION}", citation)
        self.assertEqual(CURRENT_RELEASE_VERSION, atlas_package["version"])
        self.assertEqual(CURRENT_RELEASE_VERSION, audit["release_version"])
        self.assertEqual(17, audit["supplementary_figure_outputs"])

    def test_public_terminology_contract(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        evidence = (ROOT / "atlas" / "src" / "evidence.ts").read_text(encoding="utf-8")
        terminology = (ROOT / "docs" / "terminology.md").read_text(encoding="utf-8")
        combined = readme + "\n" + evidence
        controlled = combined + "\n" + terminology
        required = (
            "Not White alone",
            "Social-context first",
            "Canopy-deficit first",
            "illustrative equal-weight",
        )
        forbidden = (
            "People of colour",
            "Social-condition first",
            "Canopy-gap first",
            "balanced ordering",
        )
        self.assertTrue(all(term in controlled for term in required))
        self.assertTrue(all(term not in combined for term in forbidden))

    def test_public_release_has_no_manuscript_revision_traces(self) -> None:
        revision_pattern = re.compile(
            r"(?<![A-Za-z0-9])v(?:3[0-9]|4[0-9])(?=$|[^A-Za-z0-9])",
            re.I,
        )
        internal_path_version_pattern = re.compile(
            r"(?:^|_)v[0-9]+(?=$|[^A-Za-z0-9])",
            re.I,
        )
        text_suffixes = {
            ".cff",
            ".csv",
            ".css",
            ".example",
            ".in",
            ".json",
            ".md",
            ".mjs",
            ".nginx",
            ".ps1",
            ".py",
            ".service",
            ".toml",
            ".ts",
            ".tsx",
            ".txt",
            ".yaml",
            ".yml",
        }
        failures = []
        for path in project_files("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if revision_pattern.search(relative) or internal_path_version_pattern.search(relative):
                failures.append(f"path: {relative}")
            if path.suffix.lower() in text_suffixes:
                text = path.read_text(encoding="utf-8", errors="replace")
                if revision_pattern.search(text) or internal_path_version_pattern.search(text):
                    failures.append(f"content: {relative}")
        self.assertEqual([], failures)

    def test_public_release_has_no_internal_review_markup(self) -> None:
        forbidden = (
            "reviewer" + "_confidence",
            "strict_" + "reviewer_audit",
            "current manuscript" + " figure",
            "nature upgrade" + " step",
            "qa_" + "completeness",
            "required_manuscript_" + "revision",
            "pass_with_required_claim_" + "revision",
            "protocol_" + "amendment",
            "manuscript_ready_" + "replacement",
            "not_a_filler_" + "replacement",
            "analysis_" + "confidence",
            "interpretive_" + "confidence",
        )
        failures = []
        for path in project_files("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".json", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            for token in forbidden:
                if token in text:
                    failures.append(f"{path.relative_to(ROOT)}: {token}")
        self.assertEqual([], failures)

    def test_code_manifest_matches_upload_candidates(self) -> None:
        manifest_path = ROOT / "manifests" / "code_manifest_sha256.csv"
        self.assertTrue(manifest_path.is_file())
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        represented = {row["relative_path"] for row in rows}
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        candidates = {
            line.replace("\\", "/")
            for line in result.stdout.splitlines()
            if line and line.replace("\\", "/") != "manifests/code_manifest_sha256.csv"
        }
        self.assertEqual(candidates, represented)
        mismatches = []
        for row in rows:
            path = ROOT / row["relative_path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["sha256"] or path.stat().st_size != int(row["bytes"]):
                mismatches.append(row["relative_path"])
        self.assertEqual([], mismatches)

    def test_portable_path_contract(self) -> None:
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from uscanopy.paths import repo_root

            self.assertEqual(ROOT.resolve(), repo_root())
        finally:
            sys.path.pop(0)

    def test_readme_links_resolve(self) -> None:
        for relative in (
            "docs/data_contract.md",
            "docs/workflow.md",
            "docs/terminology.md",
            "docs/reproducibility_scope.md",
            "docs/release_plan.md",
            "DATA_LICENSE.md",
            "CITATION.cff",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_model_contract_is_declared(self) -> None:
        source = (ROOT / "src" / "uscanopy" / "segmentation.py").read_text(encoding="utf-8")
        for required in ("class UNet", "MorphoAttention1", "DirectionCentralityModule", "self.sigmoid"):
            self.assertIn(required, source)

    def test_atlas_catalogue_matches_frozen_city_registry(self) -> None:
        atlas = ROOT / "atlas"
        with (atlas / "config" / "city_registry.csv").open(newline="", encoding="utf-8") as stream:
            registry = list(csv.DictReader(stream))
        cities = json.loads((atlas / "public" / "catalog" / "cities.json").read_text(encoding="utf-8"))
        boundaries = json.loads(
            (atlas / "public" / "catalog" / "city_boundaries.geojson").read_text(encoding="utf-8")
        )["features"]

        registry_slugs = {row["slug"] for row in registry}
        city_slugs = {city["slug"] for city in cities}
        boundary_slugs = {feature["properties"]["slug"] for feature in boundaries}
        self.assertEqual(50, len(registry))
        self.assertEqual(list(range(1, 51)), sorted(int(row["rank"]) for row in registry))
        self.assertEqual(registry_slugs, city_slugs)
        self.assertEqual(registry_slugs, boundary_slugs)
        self.assertTrue(all(city["raster"]["file"] == f"{city['slug']}.tif" for city in cities))
        self.assertEqual(11_984, sum(int(city["tracts"]) for city in cities))
        self.assertTrue(all(int(city["tracts"]) > 0 for city in cities))

    def test_atlas_neighbourhood_release_matches_frozen_typology(self) -> None:
        atlas = ROOT / "atlas"
        layer_root = atlas / "public" / "neighbourhoods"
        summary = json.loads((layer_root / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((layer_root / "release_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual("GHV27-2026-08-09", summary["freezeId"])
        self.assertEqual(34_566, summary["totalNeighbourhoods"])
        self.assertEqual(50, len(summary["cities"]))
        self.assertEqual(34_566, manifest["totalFeatures"])
        self.assertEqual(50, len(manifest["cityFiles"]))
        self.assertEqual(
            34_566,
            sum(city["features"] for city in manifest["cityFiles"]),
        )
        self.assertTrue(
            all((layer_root / city["file"]).is_file() for city in manifest["cityFiles"])
        )

    def test_primary_atlas_interface_does_not_use_legacy_hvcg(self) -> None:
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "atlas" / "src").glob("*.tsx")
        )
        self.assertNotIn("HVCG", sources.upper())

    def test_atlas_display_contract_guards_known_regressions(self) -> None:
        map_view = (ROOT / "atlas" / "src" / "MapView.tsx").read_text(encoding="utf-8")
        app = (ROOT / "atlas" / "src" / "App.tsx").read_text(encoding="utf-8")
        styles = (ROOT / "atlas" / "src" / "styles.css").read_text(encoding="utf-8")
        server = (ROOT / "atlas" / "server" / "app.py").read_text(encoding="utf-8")

        self.assertIn('if (cities.length === 0) return "#8aa49a";', map_view)
        self.assertIn("formatPopulation(selectedCity.blockGroups)", app)
        self.assertNotIn("selectedSummary?.neighbourhoods ?? selectedCity.blockGroups", app)
        self.assertRegex(
            styles,
            r"@media \(max-width: 760px\)[\s\S]*?\.canopy-rate-legend\s*\{\s*bottom:\s*100px;",
        )
        self.assertIn('"cogSource": "remote" if COG_BASE_URL else "local"', server)
        self.assertNotIn('"cogSource": COG_BASE_URL or str(COG_ROOT)', server)


if __name__ == "__main__":
    unittest.main()
