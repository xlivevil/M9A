import multiprocessing
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from agent import bootstrap


def acquire_requirements_lock_and_touch(project_root: str, touched_path: str) -> None:
    with bootstrap.requirements_install_lock(Path(project_root), timeout_seconds=5.0):
        Path(touched_path).write_text("acquired\n", encoding="utf8")


def test_requirements_lock_path_prefers_project_venv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(bootstrap, "is_running_in_project_venv", lambda _root: True)

    assert bootstrap.requirements_lock_path(tmp_path) == tmp_path / ".venv" / bootstrap.REQUIREMENTS_LOCK


def test_requirements_lock_path_falls_back_to_debug_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(bootstrap, "is_running_in_project_venv", lambda _root: False)

    assert bootstrap.requirements_lock_path(tmp_path) == tmp_path / "debug" / bootstrap.REQUIREMENTS_LOCK


def test_requirements_lock_retries_and_releases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_attempts = iter((False, True))
    monotonic_values = iter((0.0, 0.0))
    unlock = Mock()
    sleep = Mock()
    monkeypatch.setattr(bootstrap, "try_lock_file", lambda _handle: next(lock_attempts))
    monkeypatch.setattr(bootstrap, "unlock_file", unlock)
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(bootstrap.time, "sleep", sleep)

    with bootstrap.requirements_install_lock(tmp_path, timeout_seconds=1.0):
        pass

    sleep.assert_called_once_with(0.1)
    unlock.assert_called_once()


def test_requirements_lock_timeout_stops_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monotonic_values = iter((0.0, 2.0))
    warnings: list[str] = []
    monkeypatch.setattr(bootstrap, "try_lock_file", lambda _handle: False)
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(bootstrap, "warn", lambda _root, message: warnings.append(message))

    with pytest.raises(SystemExit):
        with bootstrap.requirements_install_lock(tmp_path, timeout_seconds=1.0):
            pass

    assert len(warnings) == 1


def test_requirements_lock_serializes_processes(tmp_path: Path) -> None:
    touched_path = tmp_path / "second-process-acquired"
    process = multiprocessing.get_context("spawn").Process(
        target=acquire_requirements_lock_and_touch,
        args=(str(tmp_path), str(touched_path)),
    )

    with bootstrap.requirements_install_lock(tmp_path, timeout_seconds=5.0):
        process.start()
        time.sleep(0.3)
        assert not touched_path.exists()

    process.join(timeout=5.0)
    assert process.exitcode == 0
    assert touched_path.exists()


def test_ensure_requirements_installed_skips_index_fallback_when_local_wheels_succeed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("maafw==1.0.0\n", encoding="utf8")
    warnings: list[str] = []
    monkeypatch.setattr(bootstrap, "warn", lambda _root, message: warnings.append(message))
    monkeypatch.setattr(bootstrap, "install_from_local_wheels", lambda *_args: True)
    monkeypatch.setattr(
        bootstrap,
        "install_from_indexes",
        lambda *_args: pytest.fail("index installation should not run when local wheels succeed"),
    )

    bootstrap.ensure_requirements_installed(tmp_path, requirements)

    assert warnings == []


def test_install_from_local_wheels_command_has_no_upgrade_or_index_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("maafw==1.0.0\n", encoding="utf8")
    deps_dir = tmp_path / "deps"
    deps_dir.mkdir()
    (deps_dir / "maafw-1.0.0-py3-none-any.whl").write_bytes(b"wheel")
    commands: list[list[str]] = []

    def record_run_pip(_project_root: Path, command: list[str], _label: str) -> bool:
        commands.append(command)
        return True

    monkeypatch.setattr(bootstrap, "run_pip", record_run_pip)

    assert bootstrap.install_from_local_wheels(tmp_path, requirements) is True

    command = commands[0]
    assert "--no-index" in command
    assert "--find-links" in command
    assert not {"-U", "-i", "--extra-index-url"} & set(command)


def test_install_from_indexes_command_has_no_upgrade_or_mirror_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("maafw==1.0.0\n", encoding="utf8")
    commands: list[list[str]] = []

    def record_run_pip(_project_root: Path, command: list[str], _label: str) -> bool:
        commands.append(command)
        return True

    monkeypatch.setattr(bootstrap, "run_pip", record_run_pip)

    assert bootstrap.install_from_indexes(tmp_path, requirements) is True

    command = commands[0]
    assert "--requirement" in command
    assert not {"-U", "-i", "--extra-index-url"} & set(command)


class FakeVersionProbe:
    def __init__(self, version_output: str) -> None:
        self.returncode = 0
        self.stdout = version_output
        self.stderr = ""


def test_find_compatible_python_skips_incompatible_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probes: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> FakeVersionProbe:
        probes.append(command)
        return FakeVersionProbe("Python 3.12.10")

    monkeypatch.setattr(bootstrap.shutil, "which", lambda _name: str(tmp_path / "python3"))
    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    assert bootstrap.find_compatible_python() is None
    assert probes


def test_find_compatible_python_returns_first_matching_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(command: list[str], **_kwargs: object) -> FakeVersionProbe:
        return FakeVersionProbe("Python 3.13.5")

    monkeypatch.setattr(bootstrap.shutil, "which", lambda _name: str(tmp_path / "python3.13"))
    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    assert bootstrap.find_compatible_python() == tmp_path / "python3.13"
