"""Bounded, read-only G4-S1A license provenance collector.

Only license/documentation URLs are requested. No USD or mesh is downloaded,
converted, redistributed, or passed to a physics engine. A successful generic
license fetch never establishes its applicability to an individual USD asset.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/"
PIN = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
ISAAC_PIN = "90b79bb2d44feb8d833f260f2bf37da3487180ba"
CURRENT_TERMS = "https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/"


def file_record(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    data = path.read_bytes()
    return {"path": str(path), "exists": True, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def probe_text(url: str) -> dict:
    """GET at most 64 KiB of a predeclared license/documentation resource."""
    started = time.monotonic()
    row = {"url": url, "method": "GET", "maximum_response_bytes": 65536}
    request = urllib.request.Request(url, headers={"User-Agent": "G4-S1A-license-provenance/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = response.read(65537)
            row.update(status=response.status, final_url=response.url,
                       headers={key: response.headers.get(key) for key in
                                ("Content-Type", "Content-Length", "ETag", "Last-Modified", "Date")})
            row["response_truncated"] = len(data) > 65536
            if not row["response_truncated"]:
                row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                text = data.decode("utf-8", errors="replace")
                row["license_identification"] = (
                    "APACHE_2_0" if "Apache License" in text and "Version 2.0" in text else
                    "NVIDIA_TERMS_PAGE_NOT_ASSET_SPECIFIC" if "NVIDIA" in text else
                    "NOT_CLASSIFIED")
            # Do not save fetched documents or original asset data.
    except urllib.error.HTTPError as exc:
        row.update(status=exc.code, error=str(exc), response_body_saved=False)
    except (OSError, ValueError) as exc:
        row.update(status=None, error=f"{type(exc).__name__}: {exc}")
    row["runtime_s"] = time.monotonic() - started
    return row


def collect(network: bool = False) -> dict:
    s0 = json.loads((ROOT / "evidence/g4_s/s0/assets_audit.json").read_text())
    paths = [Path("/home/ubuntu2004/isaacsim-4.5.0/LICENSE.txt"),
             Path("/home/ubuntu2004/isaacsim-4.5.0/PACKAGE-LICENSES/isaac-sim-assets-LICENSE.txt"),
             Path("/home/ubuntu2004/isaacsim-4.5.0/PACKAGE-LICENSES/franka-LICENSE.txt"),
             Path("/home/ubuntu2004/mujoco_menagerie/franka_emika_panda/LICENSE"),
             Path("/home/ubuntu2004/mujoco_menagerie/franka_emika_panda/README.md"),
             Path("/home/ubuntu2004/mujoco_menagerie/franka_emika_panda/NOTICE"),
             Path("/home/ubuntu2004/mujoco_menagerie/NOTICE")]
    paths += [Path(row["path"]) for row in s0["license_source_files"]
              if Path(row["path"]) not in paths]
    remote = []
    urls = []
    # Direct hosted folders, known conventional names; bounded negative probes
    # are discovery evidence only and do not demonstrate absence of any license.
    for folder, names in [
        ("IsaacLab/Factory/", ["LICENSE", "LICENSE.txt", "license.txt", "franka-LICENSE.txt", "LICENSE.md"]),
        ("IsaacLab/", ["LICENSE", "LICENSE.txt"]),
        ("Robots/Franka/", ["LICENSE", "LICENSE.txt", "franka-LICENSE.txt"]),
        ("Props/Mounts/SeattleLabTable/", ["LICENSE", "LICENSE.txt", "license.txt"]),
        ("Props/", ["LICENSE", "LICENSE.txt"]),
    ]:
        urls.extend(S3 + folder + name for name in names)
    urls += [
        "https://docs.isaacsim.omniverse.nvidia.com/4.5.0/assets/usd_assets_robots.html",
        "https://docs.isaacsim.omniverse.nvidia.com/4.5.0/common/NVIDIA_Omniverse_License_Agreement.html",
        CURRENT_TERMS,
        f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{PIN}/franka_emika_panda/LICENSE",
        f"https://raw.githubusercontent.com/isaac-sim/IsaacLab/{ISAAC_PIN}/docs/licenses/assets/franka-license.txt",
    ]
    if network:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            remote = list(pool.map(probe_text, urls))
    assets = []
    for name in ["franka_mimic.usd", "factory_peg_8mm.usd", "factory_hole_8mm.usd"]:
        rec = s0["original_cached_files"][name]
        assets.append({
            "name": name, "reference_sha256": rec["sha256"],
            "source_url": S3 + "IsaacLab/Factory/" + name,
            "applicable_exact_asset_license": "UNKNOWN",
            "local_conversion_permission": "NOT_ESTABLISHED",
            "distribution_permission": "NOT_ESTABLISHED",
            "gate": "BLOCKED",
            "reason": "Bundled content-pack terms defer robot/prop licenses to hosted folders; neither the general Franka license nor repository code license establishes the scope of this exact Factory-modified USD."})
    assets.append({
        "name": "table_instanceable.usd and table.usd dependency",
        "source_url": S3 + "Props/Mounts/SeattleLabTable/table_instanceable.usd",
        "reference_sha256": "479ed8fd374d8eafdedf3ca3b9d0d5981ffc6d5c5016929347f748be93d5df17",
        "applicable_exact_asset_license": "UNKNOWN",
        "local_conversion_permission": "NOT_ESTABLISHED",
        "distribution_permission": "NOT_ESTABLISHED", "gate": "BLOCKED",
        "reason": "Hosted-folder specific license and complete dependency chain are not established; generic Content grant is not a verified per-asset clearance."})
    assets.append({
        "name": "Menagerie franka_emika_panda",
        "source_url": f"https://github.com/google-deepmind/mujoco_menagerie/tree/{PIN}/franka_emika_panda",
        "commit": PIN, "applicable_exact_asset_license": "Apache-2.0",
        "local_conversion_permission": "VERIFIED_UNDER_LICENSE_CONDITIONS",
        "distribution_permission": "VERIFIED_UNDER_LICENSE_CONDITIONS",
        "gate": "DIAGNOSTIC_ONLY_ALLOWED",
        "requirements": ["Provide Apache-2.0 license with distributed derivative copies.",
                         "Retain relevant copyright/patent/trademark/attribution notices.",
                         "Mark changed files; preserve NOTICE contents if supplied.",
                         "No NOTICE found at model/root paths in the pinned checkout; this does not waive notices elsewhere.",
                         "Do not imply this license covers NVIDIA Factory modifications, peg, hole or table."]})
    return {
        "schema_version": 1, "task": "G4-S1A", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Read-only license provenance and conservative implementation gate; not a blanket legal clearance.",
        "s0_evidence": file_record(ROOT / "evidence/g4_s/s0/assets_audit.json"),
        "collector_source": file_record(Path(__file__)),
        "local_license_sources": [file_record(path) for path in paths],
        "existing_s0_layer_license_observations": [{
            "name": asset["name"],
            "custom_layer_data": asset["custom_layer_data"],
            "layer_documentation": asset["layer_metadata"].get("documentation", ""),
            "interpretation": "No applicable asset license identified in these already-recorded layer metadata fields; this is not a complete search of all asset notices."}
            for asset in s0["original_usd_probe"]["result"]["assets"]],
        "network_enabled": network, "network_probes": remote,
        "asset_license_matrix": assets,
        "bundled_4_5_agreement_analysis": {
            "status": "VERIFIED_TEXT_CONDITIONAL_APPLICABILITY",
            "definition": "Section 1.4 defines Content as NVIDIA audio/2D/3D assets.",
            "local_use": "Section 2.1 grants installation/use subject to license limitations.",
            "local_conversion": "Section 2.1 permits modification/derivatives of Content, if the exact asset is NVIDIA Content governed by this agreement and no overriding component terms apply.",
            "distribution": "Section 2.1 permits Content as incorporated into products/services enhancing the work; standalone Content distribution is excluded; section 3 imposes additional distribution requirements.",
            "component_override": "Components Under Other Licenses allows component-specific notices to control conflicts; isaac-sim-assets-LICENSE directs per-robot/prop terms to hosted folders.",
            "unresolved": "No evidence ties each exact required Factory USD and all third-party portions to a complete applicable license scope.",
            "implementation_consequence": "No original/derived asset conversion or distribution is authorized by this audit. Existing originals remain local read-only references; only independent Apache Panda diagnostics may be compiled."},
        "current_terms_not_retroactive": {
            "url": CURRENT_TERMS, "edition": "June 9, 2025",
            "status": "CONTEXT_ONLY_NOT_APPLIED_TO_FROZEN_4_5_ASSETS",
            "reason": "The current page states a different grant/restrictions from the installed 4.5 agreement. A current web page cannot establish terms supplied with January2025 asset bytes, nor be selectively used to infer conversion permission."},
        "positive_hosted_folder_finding": {
            "url": S3 + "Robots/Franka/franka-LICENSE.txt",
            "license": "Apache-2.0",
            "scope": "The stock Robots/Franka folder; no verified attribution or dependency link extends it to Factory/Forge modifications, peg, hole or table.",
            "applicability_to_frozen_franka_mimic": "UNKNOWN"},
        "negative_probe_interpretation": "404/not-found URLs prove only that no license was found there, never that no license exists or that assets are prohibited.",
        "original_or_derived_assets_copied_to_repository": False,
        "engine_or_physics_execution": False,
        "decision": "ORIGINAL_ASSET_CONVERSION_AND_DISTRIBUTION_BLOCKED; INDEPENDENT_PANDA_DIAGNOSTICS_ALLOWED",
        "unblocking_evidence_needed": [
            "Exact applicable asset license/attribution for Factory franka_mimic modifications, peg, hole and table dependencies.",
            "Verified local conversion/cross-engine use scope, including component-specific third-party terms.",
            "Separate public redistribution basis if original or derived geometry is to be committed/distributed."]}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result = collect(args.network)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(result["decision"])


if __name__ == "__main__":
    main()
