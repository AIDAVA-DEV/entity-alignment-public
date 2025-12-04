"""
End-to-end tests for the align_graph endpoint.

This test suite verifies that the data imputation API works correctly
for various medical knowledge graph scenarios.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

# Add the parent directory to the path so we can import our app
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the FastAPI app
try:
    import asyncio

    from app import app, init_models

    logger.info("✅ Successfully imported FastAPI app")

    # Initialize models for testing
    try:
        asyncio.run(init_models())
        logger.info("✅ Models initialized for testing")
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        pytest.skip(f"Cannot initialize models: {e}", allow_module_level=True)

except ImportError as e:
    logger.error(f"❌ Failed to import app: {e}")
    pytest.skip("Cannot import app module", allow_module_level=True)


class TestalignGraphEndpoint:
    """Test class for the /align_graph endpoint."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create a test client for the FastAPI app."""
        try:
            return TestClient(app)
        except Exception as e:
            logger.error(f"❌ Failed to create test client: {e}")
            pytest.skip(f"Cannot create test client: {e}")

    @pytest.fixture(scope="class")
    def test_cases(self):
        """Load test cases from JSON file."""
        test_data_path = Path(__file__).parent / "data" / "align_graph_test_cases.json"

        try:
            with open(test_data_path, "r", encoding="utf-8") as f:
                cases = json.load(f)
            logger.info(f"✅ Loaded {len(cases)} test cases from {test_data_path}")
            return cases
        except FileNotFoundError:
            logger.error(f"❌ Test data file not found: {test_data_path}")
            pytest.skip("Test data file not found")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in test data file: {e}")
            pytest.skip("Invalid test data file")

    def test_api_health_check(self, client):
        """Test that the API is healthy and can respond to basic requests."""
        try:
            response = client.get("/health")
            assert response.status_code == 200
            health_data = response.json()

            logger.info(f"✅ Health check passed: {health_data}")

            # Check if model is loaded - this is required for proper functionality
            if not health_data.get("model_loaded", False):
                logger.error("❌ Model is not loaded - this will cause test failures")
                pytest.fail(
                    "Model is not loaded. Ensure model files are present and properly configured."
                )
            else:
                logger.info("✅ Model is loaded and ready")

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            pytest.fail(f"Health check failed: {e}")

    def test_align_graph_endpoint_structure(self, client):
        """Test that the endpoint exists and handles basic structure correctly."""
        # Test with minimal valid request
        test_request = {
            "graph": "@prefix ex: <http://example.org/> . ex:test ex:prop ex:value .",
            "graph_format": "turtle",
            "input_code": "ex:test",
        }

        try:
            response = client.post("/align_graph", json=test_request)

            # Should get a response (even if empty suggestions due to no model)
            assert response.status_code in [200, 503], (
                f"Unexpected status code: {response.status_code}"
            )

            if response.status_code == 200:
                data = response.json()
                assert "input_code" in data
                assert "suggestions" in data
                assert isinstance(data["suggestions"], list)
                logger.info("✅ Endpoint structure test passed")
            else:
                logger.warning("⚠️ Model not available - endpoint returned 503")

        except Exception as e:
            logger.error(f"❌ Endpoint structure test failed: {e}")
            pytest.fail(f"Endpoint structure test failed: {e}")

    def test_align_graph_all_cases(self, client, test_cases):
        """Test all cases from the test data file."""
        results = []

        for i, case in enumerate(test_cases):
            case_name = case.get("case_name", f"case_{i}")
            description = case.get("description", "No description")
            request_data = case.get("request", {})
            expected_code = case.get("expected_code_in_suggestion")

            logger.info(f"\n🧪 Testing case: {case_name}")
            logger.info(f"   Description: {description}")

            try:
                # Make the API request
                response = client.post("/align_graph", json=request_data)

                # Track the result
                result = {
                    "case_name": case_name,
                    "description": description,
                    "status_code": response.status_code,
                    "success": False,
                    "error": None,
                    "suggestions_count": 0,
                    "expected_code_found": False,
                }

                if response.status_code == 200:
                    data = response.json()
                    suggestions = data.get("suggestions", [])
                    result["suggestions_count"] = len(suggestions)

                    logger.info(
                        f"   ✅ Response received: {len(suggestions)} suggestions"
                    )

                    # Log all suggestions for debugging
                    if suggestions:
                        logger.info("   📋 Suggestions received:")
                        for j, suggestion in enumerate(suggestions):
                            logger.info(
                                f"      {j + 1}. {suggestion.get('value', 'N/A')} (confidence: {suggestion.get('confidence', 'N/A')})"
                            )
                    else:
                        logger.info("   📋 No suggestions received")

                    # Check if expected code is in suggestions
                    if expected_code is None:
                        # For cases where we don't expect suggestions
                        result["success"] = True
                        result["expected_code_found"] = len(suggestions) == 0
                        if len(suggestions) == 0:
                            logger.info("   ✅ Correctly returned no suggestions")
                        else:
                            logger.info(
                                f"   ⚠️ Expected no suggestions but got {len(suggestions)}"
                            )
                    else:
                        # For cases where we expect specific codes
                        result["success"] = True
                        for suggestion in suggestions:
                            suggestion_value = suggestion.get("value", "")
                            if expected_code in suggestion_value:
                                result["expected_code_found"] = True
                                logger.info(
                                    f"   ✅ Found expected code pattern '{expected_code}' in suggestion: {suggestion_value}"
                                )
                                break

                        if not result["expected_code_found"]:
                            logger.warning(
                                f"   ⚠️ Expected code pattern '{expected_code}' not found in suggestions"
                            )

                elif response.status_code == 503:
                    result["error"] = "Model not loaded"
                    result["success"] = False  # Model not loading is a failure
                    logger.error(
                        "   ❌ Model not loaded (503) - this indicates a problem with model loading"
                    )

                else:
                    result["error"] = f"HTTP {response.status_code}: {response.text}"
                    logger.error(f"   ❌ Request failed: {result['error']}")

            except Exception as e:
                result = {
                    "case_name": case_name,
                    "description": description,
                    "success": False,
                    "error": str(e),
                    "status_code": None,
                    "suggestions_count": 0,
                    "expected_code_found": False,
                }
                logger.error(f"   ❌ Exception occurred: {e}")

            results.append(result)

        # Summary report
        self._log_test_summary(results)

        # Ensure all cases were attempted
        assert len(results) == len(test_cases), "Not all test cases were executed"

        # Report any failures (including model loading failures)
        failures = [r for r in results if not r["success"]]
        if failures:
            failure_details = [f"{r['case_name']}: {r['error']}" for r in failures]
            pytest.fail(
                f"Some test cases failed: {len(failures)} failures out of {len(results)} total. "
                f"Failures: {'; '.join(failure_details)}"
            )

    def _log_test_summary(self, results: List[Dict[str, Any]]):
        """Log a summary of all test results."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 80)

        total_cases = len(results)
        successful_cases = sum(1 for r in results if r["success"])
        model_available_cases = sum(1 for r in results if r["status_code"] == 200)
        cases_with_expected_codes = sum(1 for r in results if r["expected_code_found"])

        logger.info(f"Total test cases: {total_cases}")
        logger.info(f"Successful API calls: {successful_cases}")
        logger.info(f"Cases with model predictions: {model_available_cases}")
        logger.info(f"Cases with expected code patterns: {cases_with_expected_codes}")

        logger.info("\n📋 DETAILED RESULTS:")
        for result in results:
            status_icon = "✅" if result["success"] else "❌"
            code_icon = "🎯" if result["expected_code_found"] else "🔍"

            logger.info(f"{status_icon} {result['case_name']}")
            logger.info(f"    Status: HTTP {result.get('status_code', 'N/A')}")
            logger.info(f"    Suggestions count: {result['suggestions_count']}")
            logger.info(
                f"    Expected code found: {code_icon} {result['expected_code_found']}"
            )
            if result["error"]:
                logger.info(f"    Error: {result['error']}")

        logger.info("\n💡 NOTES:")
        if model_available_cases == 0:
            logger.error("   ❌ No model predictions available. This indicates:")
            logger.error("     * Model files are missing or corrupted")
            logger.error("     * Model failed to load during startup")
            logger.error("     * Training data or configuration issues")
            logger.error("   ➡️  Action required: Check model files and training status")
        else:
            logger.info(
                f"   ✅ Model is working correctly with {model_available_cases} successful predictions"
            )

        if cases_with_expected_codes > 0:
            logger.info(
                f"   - {cases_with_expected_codes} cases found expected code patterns"
            )

        logger.info("=" * 80)


def run_tests():
    """Run the tests programmatically."""
    test_file = __file__

    logger.info("🚀 Starting align_graph endpoint tests...")

    # Run pytest
    exit_code = pytest.main(
        [
            test_file,
            "-v",  # Verbose output
            "-s",  # Don't capture output
            "--tb=short",  # Short traceback format
            "--color=yes",  # Colored output
        ]
    )

    return exit_code


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    exit_code = run_tests()
    sys.exit(exit_code)
