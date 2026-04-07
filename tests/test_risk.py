"""
Unit tests for risk_classifier.classify_risk() function.

Tests cover all classification rules and edge cases.
"""

import pytest
import math
from risk_classifier import classify_risk


@pytest.fixture
def base_row():
    """
    Fixture providing a baseline row with neutral/normal values.
    
    Can be overridden per test by copying and updating specific keys.
    """
    return {
        "anomaly_flag": False,
        "anomaly_score": 0.5,
        "in_regional_cluster": False,
        "zscore_category": "normal",
        "rolling_zscore": 1.2,
        "dual_source": False,
        "rainfall_gov": 25.0,
        "rainfall_meteo": math.nan,  # Single source case
    }


class TestNormalCase:
    """Test: No anomaly detected, z-score < 2, single source."""

    def test_normal_case(self, base_row):
        """Scenario: Normal rainfall with no anomalies → Normal, High confidence."""
        result = classify_risk(base_row)

        assert result["risk_level"] == "Normal"
        assert result["confidence"] == "High"
        assert result["reason"] == "No anomaly detected"
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


class TestModerateIsolation:
    """Test: Isolated anomaly with moderate Z-score."""

    def test_moderate_isolation_only(self, base_row):
        """
        Scenario: Point anomaly detected but not in cluster,
        with moderate Z-score → Moderate Risk, Medium confidence.
        """
        base_row["anomaly_flag"] = True
        base_row["zscore_category"] = "moderate"
        base_row["rolling_zscore"] = 2.2

        result = classify_risk(base_row)

        assert result["risk_level"] == "Moderate Risk"
        assert result["confidence"] == "Medium"
        assert "Isolated anomaly" in result["reason"]
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


class TestHighRisk:
    """Test: Regional anomaly cluster with extreme Z-score (single source)."""

    def test_high_risk(self, base_row):
        """
        Scenario: Anomaly in regional cluster with extreme Z-score,
        but single data source → High Risk, High confidence.
        """
        base_row["anomaly_flag"] = True
        base_row["in_regional_cluster"] = True
        base_row["zscore_category"] = "extreme"
        base_row["rolling_zscore"] = 3.5
        base_row["dual_source"] = False

        result = classify_risk(base_row)

        assert result["risk_level"] == "High Risk"
        assert result["confidence"] == "High"
        assert "Regional anomaly cluster" in result["reason"]
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


class TestCriticalRisk:
    """Test: Regional anomaly confirmed by dual sources."""

    def test_critical_risk(self, base_row):
        """
        Scenario: Regional cluster + extreme Z-score + confirmed by dual sources
        → Critical Risk, Very High confidence.
        """
        base_row["anomaly_flag"] = True
        base_row["in_regional_cluster"] = True
        base_row["zscore_category"] = "extreme"
        base_row["rolling_zscore"] = 4.2
        base_row["dual_source"] = True
        base_row["rainfall_gov"] = 150.0
        base_row["rainfall_meteo"] = 152.0

        result = classify_risk(base_row)

        assert result["risk_level"] == "Critical Risk"
        assert result["confidence"] == "Very High"
        assert "dual sources" in result["reason"]
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


class TestDualSourceDisagreement:
    """Test: Dual sources present but disagree by >50 mm."""

    def test_dual_source_disagreement(self, base_row):
        """
        Scenario: Both rainfall sources available but disagree by >50 mm
        (data quality issue) → Moderate Risk, Low confidence.
        """
        base_row["dual_source"] = True
        base_row["rainfall_gov"] = 10.0
        base_row["rainfall_meteo"] = 80.0  # 70 mm difference

        result = classify_risk(base_row)

        assert result["risk_level"] == "Moderate Risk"
        assert result["confidence"] == "Low"
        assert "Dual-source disagreement" in result["reason"]
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


class TestReasonAlwaysString:
    """Test: All classification paths return non-empty reason strings."""

    def test_reason_is_string_normal(self, base_row):
        """Reason field is always a non-empty string (normal case)."""
        result = classify_risk(base_row)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_reason_is_string_moderate_isolation(self, base_row):
        """Reason field is always a non-empty string (moderate isolation case)."""
        base_row["anomaly_flag"] = True
        base_row["zscore_category"] = "moderate"
        result = classify_risk(base_row)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_reason_is_string_high_risk(self, base_row):
        """Reason field is always a non-empty string (high risk case)."""
        base_row["anomaly_flag"] = True
        base_row["in_regional_cluster"] = True
        base_row["zscore_category"] = "extreme"
        base_row["dual_source"] = False
        result = classify_risk(base_row)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_reason_is_string_critical(self, base_row):
        """Reason field is always a non-empty string (critical risk case)."""
        base_row["anomaly_flag"] = True
        base_row["in_regional_cluster"] = True
        base_row["zscore_category"] = "extreme"
        base_row["dual_source"] = True
        base_row["rainfall_gov"] = 150.0
        base_row["rainfall_meteo"] = 152.0
        result = classify_risk(base_row)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_reason_is_string_disagreement(self, base_row):
        """Reason field is always a non-empty string (disagreement case)."""
        base_row["dual_source"] = True
        base_row["rainfall_gov"] = 10.0
        base_row["rainfall_meteo"] = 80.0
        result = classify_risk(base_row)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_reason_is_string_default(self, base_row):
        """Reason field is always a non-empty string (default fallback case)."""
        # Create a row that doesn't match any explicit rule
        base_row["anomaly_flag"] = True
        base_row["zscore_category"] = "moderate"
        base_row["in_regional_cluster"] = True
        result = classify_risk(base_row)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_nan_handling_disagreement_not_triggered(self, base_row):
        """
        Scenario: One source is NaN, other is numeric.
        Should not trigger disagreement rule.
        """
        base_row["dual_source"] = True
        base_row["rainfall_gov"] = 50.0
        base_row["rainfall_meteo"] = math.nan

        result = classify_risk(base_row)

        # Should not trigger disagreement rule (NaN skips check)
        assert result["risk_level"] in ["Normal", "Moderate Risk"]
        assert isinstance(result["reason"], str)

    def test_exact_threshold_moderate_z(self, base_row):
        """
        Scenario: Z-score exactly at threshold (z=2.0).
        Should be considered moderate, not extreme.
        """
        base_row["anomaly_flag"] = True
        base_row["zscore_category"] = "moderate"
        base_row["rolling_zscore"] = 2.0

        result = classify_risk(base_row)

        assert result["risk_level"] == "Moderate Risk"

    def test_exact_threshold_disagreement(self, base_row):
        """
        Scenario: Disagreement exactly at 50mm threshold.
        Should NOT trigger (needs > 50mm).
        """
        base_row["dual_source"] = True
        base_row["rainfall_gov"] = 100.0
        base_row["rainfall_meteo"] = 50.0  # Exactly 50mm difference

        result = classify_risk(base_row)

        # Exactly 50mm should not trigger disagreement (needs > 50)
        assert result["risk_level"] != "Moderate Risk" or "disagreement" not in result["reason"].lower()


class TestReturnStructure:
    """Test that return values always have required structure."""

    def test_return_has_all_keys(self, base_row):
        """Return dict always has all required keys."""
        result = classify_risk(base_row)

        required_keys = {"risk_level", "confidence", "reason"}
        assert required_keys.issubset(result.keys())
        assert len(result) == 3

    def test_return_values_are_strings(self, base_row):
        """All return values are strings."""
        result = classify_risk(base_row)

        assert isinstance(result["risk_level"], str)
        assert isinstance(result["confidence"], str)
        assert isinstance(result["reason"], str)

    def test_risk_level_is_valid(self, base_row):
        """risk_level is one of the four valid categories."""
        valid_levels = {"Normal", "Moderate Risk", "High Risk", "Critical Risk"}
        result = classify_risk(base_row)
        assert result["risk_level"] in valid_levels

    def test_confidence_is_valid(self, base_row):
        """confidence is one of the four valid categories."""
        valid_confidences = {"Low", "Medium", "High", "Very High"}
        result = classify_risk(base_row)
        assert result["confidence"] in valid_confidences
