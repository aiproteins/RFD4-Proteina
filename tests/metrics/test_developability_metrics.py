"""Tests for developability metrics: sequence (GRAVY, instability) and Rosetta scoring."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from atomworks.enums import ChainType
from atomworks.io.parser import parse
from atomworks.io.tools.inference import sequence_to_annotated_atom_array
from biotite.structure import AtomArray, concatenate

import rfproteina.metrics.physics.developability_metrics as developability_metrics
from rfproteina.metrics.physics.developability_metrics import (
    RosettaDevelopabilityMetric,
    SequenceDevelopabilityMetric,
    _host_python_with_pyrosetta,
    compute_gravy,
    compute_instability_index,
    protein_one_letter_sequence,
    subset_to_designed_protein,
)

# Ubiquitin (1UBQ) — accuracy anchor against published ProtParam values.
_UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

# GRAVY threshold from configs/metrics/pass_criteria: soluble designs pass, membrane fail.
_GRAVY_MAX = 0.1
_SOLUBLE = {
    "ubiquitin_1ubq": _UBQ,
    "lysozyme_6lyz": (
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCND"
        "GRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    ),
}
_MEMBRANE = {
    "rhodopsin_1f88": (
        "MNGTEGPNFYVPFSNKTGVVRSPFEAPQYYLAEPWQFSMLAAYMFLLIMLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVFGGFTT"
        "TLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLVGWSRYIPEGMQCS"
        "CGIDYYTPHEETNNESFVIYMFVVHFIIPLIVIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIAFLICWLPYAGVAFYIFTHQ"
        "GSDFGPIFMTIPAFFAKTSAVYNPVIYIMMNKQFRNCMVTTLCCGKNPLGDDEASTTVSKTETSQVAPA"
    ),
    "aquaporin_z_1rc2": (
        "MFRKLAAECFGTFWLVFGGCGSAVLAAGFPELGIGFAGVALAFGLTVLTMAFAVGHISGGHFNPAVTIGLWAGGRFPAKEVVGYVIAQVVGGI"
        "VAAALLYLIASGKTGFDAAASGFASNGYGEHSPGGYSMLSALVVELVLSAGFLLVIHGATDKFAPAGFAPIAIGLALTLIHLISIPVTNTSVN"
        "PARSTAVAIFQGGWALEQLWFFWVVPIVGGIIGGLIYRTLLEKRD"
    ),
}


def _atom_array_from_sequence(sequence: str, chain_id: str = "A") -> AtomArray:
    """Build an annotated protein AtomArray from a one-letter sequence (via atomworks)."""
    return sequence_to_annotated_atom_array(sequence, chain_id, chain_type=ChainType.POLYPEPTIDE_L)


def test_matches_protparam_ubiquitin():
    """GRAVY/instability match published ProtParam values, catching table typos."""
    assert compute_gravy(_UBQ) == pytest.approx(-0.489, abs=1e-2)
    assert compute_instability_index(_UBQ) == pytest.approx(36.06, abs=0.1)


def test_compute_gravy_normalizes_and_validates():
    assert compute_gravy(" acde ") == pytest.approx(compute_gravy("ACDE"))
    with pytest.raises(ValueError):
        compute_gravy("ACDEZ")


@pytest.mark.parametrize("sequence", _SOLUBLE.values(), ids=_SOLUBLE.keys())
def test_soluble_proteins_pass_gravy(sequence: str):
    assert compute_gravy(sequence) <= _GRAVY_MAX


@pytest.mark.parametrize("sequence", _MEMBRANE.values(), ids=_MEMBRANE.keys())
def test_membrane_proteins_fail_gravy(sequence: str):
    assert compute_gravy(sequence) > _GRAVY_MAX


def test_sequence_metric_reports_only_gravy_and_instability():
    result = SequenceDevelopabilityMetric().compute(_atom_array_from_sequence(_UBQ))
    assert set(result) == {"gravy", "instability_index"}
    assert result["gravy"] == pytest.approx(compute_gravy(_UBQ))
    assert result["instability_index"] == pytest.approx(compute_instability_index(_UBQ))


def test_sequence_metric_empty_is_nan():
    result = SequenceDevelopabilityMetric().compute(AtomArray(0))
    assert np.isnan(result["gravy"]) and np.isnan(result["instability_index"])


def test_subset_to_designed_chain_selects_chain_of_interest():
    complex_arr = concatenate(
        [_atom_array_from_sequence("ACDEFGHIKL", "A"), _atom_array_from_sequence("MNPQRSTVWY", "B")]
    )
    designed = subset_to_designed_protein(complex_arr, chain_of_interest="A")
    assert protein_one_letter_sequence(designed) == "ACDEFGHIKL"


def _pin_no_host_pyrosetta(monkeypatch):
    """Force the container route: no $PYROSETTA_PYTHON, no importable pyrosetta here.

    Without this, the apptainer assertions below silently describe the native route on any
    machine that happens to have pyrosetta installed.
    """
    monkeypatch.delenv("PYROSETTA_PYTHON", raising=False)
    monkeypatch.setattr(developability_metrics, "_host_python_with_pyrosetta", lambda: None)


def test_host_python_detection_reports_this_interpreter_only_when_importable(monkeypatch):
    monkeypatch.setattr(developability_metrics.importlib.util, "find_spec", lambda _name: object())
    assert developability_metrics._host_python_with_pyrosetta() == sys.executable
    monkeypatch.setattr(developability_metrics.importlib.util, "find_spec", lambda _name: None)
    assert developability_metrics._host_python_with_pyrosetta() is None


def test_rosetta_metric_without_any_pyrosetta_route_raises(monkeypatch):
    monkeypatch.delenv("PYROSETTA_CONTAINER", raising=False)
    _pin_no_host_pyrosetta(monkeypatch)
    metric = RosettaDevelopabilityMetric()
    with pytest.raises(OSError, match="PyRosetta not available"):
        metric.compute(AtomArray(0))


def test_rosetta_metric_assembles_command_and_reads_json(tmp_path, monkeypatch):
    """The metric shells the apptainer command with the right flags and parses its JSON."""
    _pin_no_host_pyrosetta(monkeypatch)
    container = tmp_path / "pyrosetta.sif"
    container.touch()

    def fake_run(command, **kwargs):
        script = Path(command[command.index("python") + 1])
        assert script.is_file()
        assert script.parent.name == "physics"
        assert f"{script.parent}:{script.parent}:ro" in command
        assert {"--cif", "--sasa_cutoff", "--output_json"} <= set(command)
        out_path = command[command.index("--output_json") + 1]
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump({"sap_score": 12.5, "hydrophobic_exposure_sasa": 345.0}, handle)
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "rfproteina.utils.structure_export.write_input_structure_cif",
        lambda _arr, path: path.write_text("data_design\n", encoding="utf-8"),
    )
    monkeypatch.setattr("rfproteina.metrics.physics.developability_metrics.subprocess.run", fake_run)

    metric = RosettaDevelopabilityMetric(
        container=str(container),
        apptainer_bin="apptainer",
        binds=[],
        scratch_dir=str(tmp_path),
    )
    assert metric.compute(_atom_array_from_sequence("ACDE")) == {
        "sap_score": 12.5,
        "hydrophobic_exposure_sasa": 345.0,
    }


def test_rosetta_metric_runs_host_python_when_pyrosetta_importable(tmp_path, monkeypatch):
    """With pyrosetta in the run env the helper is invoked directly -- no apptainer, same flags."""
    host_python = str(tmp_path / "envs" / "rfd4p" / "bin" / "python")
    monkeypatch.delenv("PYROSETTA_PYTHON", raising=False)
    monkeypatch.setattr(developability_metrics, "_host_python_with_pyrosetta", lambda: host_python)

    def fake_run(command, **kwargs):
        assert command[0] == host_python
        assert "exec" not in command and not any("apptainer" in arg for arg in command)
        assert "--bind" not in command
        script = Path(command[1])
        assert script.is_file() and script.name == "score_pose_rosetta.py"
        assert command[2] == "--cif" and Path(command[3]).name == "design.cif"
        assert command[command.index("--sasa_cutoff") + 1] == "20.0"
        out_path = command[command.index("--output_json") + 1]
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump({"sap_score": 12.5, "hydrophobic_exposure_sasa": 345.0}, handle)
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "rfproteina.utils.structure_export.write_input_structure_cif",
        lambda _arr, path: path.write_text("data_design\n", encoding="utf-8"),
    )
    monkeypatch.setattr("rfproteina.metrics.physics.developability_metrics.subprocess.run", fake_run)

    # A container set alongside an importable pyrosetta must not win -- that is what makes a
    # stale $PYROSETTA_CONTAINER on a cluster harmless.
    stale_container = tmp_path / "missing.sif"
    metric = RosettaDevelopabilityMetric(container=str(stale_container), scratch_dir=str(tmp_path))
    assert metric.compute(_atom_array_from_sequence("ACDE")) == {
        "sap_score": 12.5,
        "hydrophobic_exposure_sasa": 345.0,
    }


# --- Real Rosetta SAP scoring (requires a real PyRosetta: env install or apptainer) ---

_ROSETTA_SAP_MAX = 50.0  # configs/metrics/pass_criteria threshold
# Real PDB structures with SAP verified against the container (atomworks parse, full-atom):
# a soluble monomer passes, a hydrophobic membrane protein fails. Values (~): 1pgb 34, 1c3w 334.
_SAP_CASES = {"1pgb": True, "1c3w": False}
_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "developability"
_CONTAINER = os.environ.get("PYROSETTA_CONTAINER")
_HAVE_CONTAINER = shutil.which("apptainer") is not None and _CONTAINER is not None and Path(_CONTAINER).is_file()
# Either route scores for real: pyrosetta in this env (the cluster case) or the apptainer image.
_HAVE_ROSETTA = _host_python_with_pyrosetta() is not None or _HAVE_CONTAINER


@pytest.mark.skipif(not _HAVE_ROSETTA, reason="PyRosetta not available (no env install, no container)")
@pytest.mark.parametrize("pdb_id, should_pass", _SAP_CASES.items(), ids=_SAP_CASES.keys())
def test_rosetta_sap_pass_fail_on_real_structures(pdb_id: str, should_pass: bool, tmp_path):
    """Soluble monomers pass the SAP threshold; a hydrophobic membrane protein fails.

    Skipped unless pyrosetta is importable here or the apptainer ($PYROSETTA_CONTAINER) exists.
    """
    atom_array = parse(_DATA_DIR / f"{pdb_id}.cif.gz")["asym_unit"][0]
    metric = RosettaDevelopabilityMetric(scratch_dir=str(tmp_path))
    sap = metric.compute(atom_array)["sap_score"]
    assert (sap <= _ROSETTA_SAP_MAX) == should_pass
