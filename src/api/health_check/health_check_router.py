"""
Simple health check router with test integration (no extra dependencies).
"""

import asyncio
import importlib.util
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, status

router = APIRouter()
startup_time = time.time()

# Cache test results
_test_cache: Dict[str, Optional[Any]] = {
    "last_run": None,
    "results": None,
}


def parse_pytest_output(output: str) -> Dict[str, Any]:
    """
    Parse pytest output to extract test results.
    """
    if not output or not output.strip():
        return {
            "status": "error",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": 0,
            "message": "No output from pytest",
        }

    passed_match = re.search(r"(\d+) passed", output)
    failed_match = re.search(r"(\d+) failed", output)
    skipped_match = re.search(r"(\d+) skipped", output)
    duration_match = re.search(r"in ([\d.]+)s", output)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    duration = float(duration_match.group(1)) if duration_match else 0

    total = passed + failed + skipped

    if total == 0:
        status = "no_tests"
    elif failed > 0:
        status = "failed"
    else:
        status = "passed"

    return {
        "status": status,
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "duration": round(duration, 2),
    }


async def run_tests_async(test_path: str = "tests/") -> Dict[str, Any]:
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    def run_pytest_sync():
        try:
            cmd = [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=None,
            )
            output = result.stdout + result.stderr

            return {"output": output, "returncode": result.returncode, "error": None}

        except subprocess.TimeoutExpired:
            return {
                "output": "",
                "returncode": -1,
                "error": "Test execution timed out after 5 minutes",
            }
        except FileNotFoundError as e:
            return {
                "output": "",
                "returncode": -1,
                "error": f"pytest not found: {str(e)}",
            }
        except Exception as e:
            import traceback

            return {
                "output": "",
                "returncode": -1,
                "error": f"Test execution failed: {str(e)}",
                "traceback": traceback.format_exc(),
            }

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, run_pytest_sync)

    if result["error"]:
        return {
            "status": "error",
            "message": result["error"],
            "traceback": result.get("traceback", ""),
        }

    output = result["output"]

    if not output.strip() and result["returncode"] != 0:
        return {
            "status": "error",
            "message": "pytest executed but produced no output",
            "exit_code": result["returncode"],
            "hint": "Check if pytest is installed and tests exist in the specified path",
        }

    parsed_results = parse_pytest_output(output)
    parsed_results["exit_code"] = result["returncode"]

    if parsed_results.get("status") in ["failed", "no_tests"]:
        parsed_results["debug_output"] = (
            output[-1000:] if len(output) > 1000 else output
        )

    return parsed_results


def check_dependencies() -> Dict[str, str]:
    """Quick dependency health checks using importlib."""
    checks = {}

    if importlib.util.find_spec("face_recognition") is not None:
        checks["face_recognition"] = "✓"
    else:
        checks["face_recognition"] = "✗ missing"

    if importlib.util.find_spec("PIL") is not None:
        checks["PIL"] = "PIL PASSED"
    else:
        checks["PIL"] = "PIL missing"

    # Check numpy
    if importlib.util.find_spec("numpy") is not None:
        checks["numpy"] = "NUMPY PASSED"
    else:
        checks["numpy"] = "NUMPY missing"

    # Check fastapi
    if importlib.util.find_spec("fastapi") is not None:
        checks["fastapi"] = "FASTAPI PASSED"
    else:
        checks["fastapi"] = "FASTAPI missing"

    return checks


@router.get("/health/status", tags=["Health"], status_code=status.HTTP_200_OK)
async def detailed_status() -> Dict[str, Any]:
    """
    Basic health status endpoint.
    """
    uptime_seconds = time.time() - startup_time
    dependencies = check_dependencies()
    all_deps_ok = all("PASSED" in v for v in dependencies.values())

    return {
        "status": "healthy" if all_deps_ok else "degraded",
        "service": "Facial Recognition Microservice",
        "author": "Rogationist Computer Society",
        "version": "1.0.0",
        "uptime_seconds": round(uptime_seconds, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": dependencies,
        "endpoints": {
            "face_verification": "operational" if all_deps_ok else "check_dependencies",
            "face_extraction": "operational" if all_deps_ok else "check_dependencies",
            "facial_registration": "operational"
            if all_deps_ok
            else "check_dependencies",
        },
        "testing": {
            "last_test_run": _test_cache["last_run"],
            "run_tests_endpoint": "/health/run-tests",
            "test_summary_endpoint": "/health/test-summary",
        },
    }


@router.get("/health/run-tests", tags=["Health"])
async def run_tests(
    force: bool = Query(False, description="Force fresh test run"),
    path: str = Query("tests/", description="Test path"),
) -> Dict[str, Any]:
    """
    Execute test suite and return results.
    """
    current_time = datetime.now(timezone.utc)

    # Check cache
    if not force and _test_cache["last_run"] is not None:
        last_run = datetime.fromisoformat(_test_cache["last_run"])
        seconds_ago = (current_time - last_run).total_seconds()

        if seconds_ago < 60 and _test_cache["results"] is not None:
            return {
                "status": "cached",
                "message": f"Using cached results from {round(seconds_ago, 1)}s ago",
                "hint": "Add ?force=true to run fresh tests",
                "cached_at": _test_cache["last_run"],
                "results": _test_cache["results"],
            }
            
    start = time.time()
    results = await run_tests_async(path)
    duration = time.time() - start

    _test_cache["last_run"] = current_time.isoformat()
    _test_cache["results"] = results

    return {
        "status": "completed",
        "executed_at": current_time.isoformat(),
        "execution_time": round(duration, 2),
        "test_results": results,
        "summary": f"{results.get('passed', 0)}/{results.get('total', 0)} tests passed",
    }


@router.get("/health/test-summary", tags=["Health"])
async def test_summary() -> Dict[str, Any]:
    """
    Get cached test results without running tests.
    """
    if _test_cache["last_run"] is None:
        return {
            "status": "not_run",
            "message": "No tests have been executed yet",
            "action": "Visit /health/run-tests to execute tests",
        }

    last_run = datetime.fromisoformat(_test_cache["last_run"])
    age_seconds = (datetime.now(timezone.utc) - last_run).total_seconds()

    results = _test_cache["results"]
    if results is None:
        results = {}

    return {
        "status": "available",
        "last_run": _test_cache["last_run"],
        "age_seconds": round(age_seconds, 2),
        "results": results,
        "summary": f"{results.get('passed', 0)}/{results.get('total', 0)} passed",
    }


@router.get("/health/full", tags=["Health"])
async def full_health_check() -> Dict[str, Any]:
    """
    Comprehensive health check with dependencies and tests.
    Automatically runs tests if not recently executed.
    """
    uptime_seconds = time.time() - startup_time
    dependencies = check_dependencies()

    if _test_cache["last_run"] is None:
        test_results = await run_tests_async()
        _test_cache["last_run"] = datetime.now(timezone.utc).isoformat()
        _test_cache["results"] = test_results
        test_status = "fresh"
    else:
        test_results = _test_cache["results"]
        if test_results is None:
            test_results = {}
        test_status = "cached"

    all_deps_ok = all("PASSED" in v for v in dependencies.values())
    tests_passed = test_results.get("status") == "passed"

    overall = "healthy" if (all_deps_ok and tests_passed) else "degraded"

    return {
        "overall_status": overall,
        "service": "Facial Recognition Microservice",
        "version": "1.0.0",
        "uptime_seconds": round(uptime_seconds, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "dependencies": {
                "status": "pass" if all_deps_ok else "fail",
                "details": dependencies,
            },
            "tests": {
                "status": test_status,
                "last_run": _test_cache["last_run"],
                "results": test_results,
            },
        },
    }
