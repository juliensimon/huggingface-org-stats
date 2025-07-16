"""
Tests for the Hugging Face Organization Statistics tool.
"""

from typing import Any, Dict
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from hf_org_stats import HFStatsCollector, main


class TestHFStatsCollector:
    """Test cases for HFStatsCollector class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.collector = HFStatsCollector()

    def test_init_without_token(self) -> None:
        """Test initialization without API token."""
        collector = HFStatsCollector()
        assert collector.api is not None
        assert collector.session is not None

    def test_init_with_token(self) -> None:
        """Test initialization with API token."""
        token = "test_token"
        collector = HFStatsCollector(token=token)
        assert collector.api is not None
        assert collector.session is not None

    def test_get_safe_attr_with_valid_attr(self) -> None:
        """Test _get_safe_attr with valid attribute."""
        mock_item = Mock()
        mock_item.test_attr = 42
        result = self.collector._get_safe_attr(mock_item, "test_attr")
        assert result == 42

    def test_get_safe_attr_with_none_value(self) -> None:
        """Test _get_safe_attr with None value."""
        mock_item = Mock()
        mock_item.test_attr = None
        result = self.collector._get_safe_attr(mock_item, "test_attr", default=0)
        assert result == 0

    def test_get_safe_attr_with_missing_attr(self) -> None:
        """Test _get_safe_attr with missing attribute."""

        class Dummy:
            pass

        dummy = Dummy()
        result = self.collector._get_safe_attr(dummy, "missing_attr", default=10)
        assert result == 10

    def test_get_item_id_with_valid_attrs(self) -> None:
        """Test _get_item_id with valid attributes."""
        mock_item = Mock()
        mock_item.id = "test_id"
        result = self.collector._get_item_id(mock_item, ["id", "modelId"])
        assert result == "test_id"

    def test_get_item_id_with_fallback_attr(self) -> None:
        """Test _get_item_id with fallback attribute."""
        mock_item = Mock()
        # Set up the mock to have modelId but not id
        mock_item.id = Mock(side_effect=AttributeError("No attribute"))
        mock_item.modelId = "test_model_id"

        # Mock hasattr to return appropriate values
        def mock_hasattr(obj: object, attr: str) -> bool:
            return attr == "modelId"

        with patch("builtins.hasattr", side_effect=mock_hasattr):
            result = self.collector._get_item_id(mock_item, ["id", "modelId"])
            assert result == "test_model_id"

    def test_get_item_id_with_no_valid_attrs(self) -> None:
        """Test _get_item_id with no valid attributes."""
        mock_item = Mock()
        # Mock hasattr to return False for all attributes
        with patch("builtins.hasattr", return_value=False):
            result = self.collector._get_item_id(mock_item, ["id", "modelId"])
            assert result is None

    @patch("hf_org_stats.HfApi")
    def test_get_organization_models_success(self, mock_api: Any) -> None:
        """Test successful model collection."""
        # Mock API response
        mock_model = Mock()
        mock_model.id = "test/model"
        mock_model.downloads = 100
        mock_model.likes = 50

        mock_api_instance = Mock()
        mock_api_instance.list_models.return_value = [mock_model]
        mock_api.return_value = mock_api_instance

        collector = HFStatsCollector()
        collector.api = mock_api_instance

        result = collector.get_organization_models("test-org")

        assert len(result) == 1
        assert result[0]["model_id"] == "test/model"
        assert result[0]["downloads"] == 100
        assert result[0]["likes"] == 50

    @patch("hf_org_stats.HfApi")
    def test_get_organization_models_exception(self, mock_api: Any) -> None:
        """Test model collection with API exception."""
        mock_api_instance = Mock()
        mock_api_instance.list_models.side_effect = Exception("API Error")
        mock_api.return_value = mock_api_instance

        collector = HFStatsCollector()
        collector.api = mock_api_instance

        result = collector.get_organization_models("test-org")
        assert result == []

    @patch("hf_org_stats.HfApi")
    def test_get_organization_datasets_success(self, mock_api: Any) -> None:
        """Test successful dataset collection."""
        # Mock API response
        mock_dataset = Mock()
        mock_dataset.id = "test/dataset"
        mock_dataset.downloads = 200
        mock_dataset.likes = 75

        mock_api_instance = Mock()
        mock_api_instance.list_datasets.return_value = [mock_dataset]
        mock_api.return_value = mock_api_instance

        collector = HFStatsCollector()
        collector.api = mock_api_instance

        result = collector.get_organization_datasets("test-org")

        assert len(result) == 1
        assert result[0]["dataset_id"] == "test/dataset"
        assert result[0]["downloads"] == 200
        assert result[0]["likes"] == 75

    @patch("hf_org_stats.HfApi")
    def test_get_organization_spaces_success(self, mock_api: Any) -> None:
        """Test successful space collection."""
        # Mock API response
        mock_space = Mock()
        mock_space.id = "test/space"
        mock_space.likes = 25

        mock_api_instance = Mock()
        mock_api_instance.list_spaces.return_value = [mock_space]
        mock_api.return_value = mock_api_instance

        collector = HFStatsCollector()
        collector.api = mock_api_instance

        result = collector.get_organization_spaces("test-org")

        assert len(result) == 1
        assert result[0]["space_id"] == "test/space"
        assert result[0]["likes"] == 25

    @patch("hf_org_stats.HfApi")
    def test_get_detailed_model_stats_success(self, mock_api: Any) -> None:
        """Test successful detailed model stats collection."""
        # Mock API responses
        mock_model_info = Mock()
        mock_model_info.downloads = 100
        mock_model_info.likes = 50

        mock_model_info_all_time = Mock()
        mock_model_info_all_time.downloads_all_time = 1000

        mock_api_instance = Mock()
        mock_api_instance.model_info.side_effect = [
            mock_model_info,
            mock_model_info_all_time,
        ]
        mock_api.return_value = mock_api_instance

        collector = HFStatsCollector()
        collector.api = mock_api_instance

        result = collector.get_detailed_model_stats("test/model")

        assert result["model_id"] == "test/model"
        assert result["downloads_all_time"] == 1000
        assert result["downloads_30d"] == 100
        assert result["likes"] == 50

    @patch("hf_org_stats.HfApi")
    def test_get_detailed_model_stats_exception(self, mock_api: Any) -> None:
        """Test detailed model stats with API exception."""
        mock_api_instance = Mock()
        mock_api_instance.model_info.side_effect = Exception("API Error")
        mock_api.return_value = mock_api_instance

        collector = HFStatsCollector()
        collector.api = mock_api_instance

        result = collector.get_detailed_model_stats("test/model")

        assert result["model_id"] == "test/model"
        assert result["downloads_all_time"] == 0
        assert result["downloads_30d"] == 0
        assert result["likes"] == 0
        assert "error" in result

    def test_save_results_csv(self) -> None:
        """Test saving results in CSV format."""
        # Create test data with correct column names
        test_data = pd.DataFrame(
            {
                "model_id": ["test/model1", "test/model2"],
                "downloads_all_time": [1000, 2000],
                "downloads_30d": [100, 200],
                "likes": [50, 75],
            }
        )

        results = {"models": test_data}

        # Mock file operations
        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            self.collector.save_results(results, "test-org", "csv")

            # Verify CSV save was called
            mock_to_csv.assert_called()

    def test_save_results_json(self) -> None:
        """Test saving results in JSON format."""
        test_data = pd.DataFrame(
            {
                "model_id": ["test/model1"],
                "downloads_all_time": [1000],
                "downloads_30d": [100],
                "likes": [50],
            }
        )

        results = {"models": test_data}

        with patch("pandas.DataFrame.to_json") as mock_to_json:
            self.collector.save_results(results, "test-org", "json")
            mock_to_json.assert_called()

    def test_save_results_excel(self) -> None:
        """Test saving results in Excel format."""
        test_data = pd.DataFrame(
            {
                "model_id": ["test/model1"],
                "downloads_all_time": [1000],
                "downloads_30d": [100],
                "likes": [50],
            }
        )

        results = {"models": test_data}

        with patch("pandas.DataFrame.to_excel") as mock_to_excel:
            self.collector.save_results(results, "test-org", "excel")
            mock_to_excel.assert_called()

    def test_generate_summary_models(self) -> None:
        """Test summary generation for models."""
        # Create test data with correct column names
        test_data = pd.DataFrame(
            {
                "model_id": ["test/model1", "test/model2"],
                "downloads_all_time": [1000, 2000],
                "downloads_30d": [100, 200],
                "likes": [50, 75],
            }
        )

        results = {"models": test_data}

        summary = self.collector.generate_summary(results, "test-org")

        assert "test-org" in summary
        assert "2 total" in summary
        assert "3,000" in summary  # Total all-time downloads (with comma)
        assert "300" in summary  # Total 30-day downloads
        assert "125" in summary  # Total likes

    def test_generate_summary_datasets(self) -> None:
        """Test summary generation for datasets."""
        test_data = pd.DataFrame(
            {
                "dataset_id": ["test/dataset1", "test/dataset2"],
                "downloads_all_time": [500, 1500],
                "downloads_30d": [50, 150],
                "likes": [25, 100],
            }
        )

        results = {"datasets": test_data}

        summary = self.collector.generate_summary(results, "test-org")

        assert "test-org" in summary
        assert "2 total" in summary
        assert "2,000" in summary  # Total all-time downloads (with comma)
        assert "200" in summary  # Total 30-day downloads
        assert "125" in summary  # Total likes

    def test_generate_summary_spaces(self) -> None:
        """Test summary generation for spaces."""
        test_data = pd.DataFrame(
            {"space_id": ["test/space1", "test/space2"], "likes": [10, 40]}
        )

        results = {"spaces": test_data}

        summary = self.collector.generate_summary(results, "test-org")

        assert "test-org" in summary
        assert "2 total" in summary
        assert "50" in summary  # Total likes

    def test_generate_summary_empty_results(self) -> None:
        """Test summary generation with empty results."""
        results: Dict[str, pd.DataFrame] = {}
        summary = self.collector.generate_summary(results, "test-org")

        assert "test-org" in summary
        assert "===" in summary

    @patch("hf_org_stats.HfApi")
    def test_collect_organization_stats_models_only(self, mock_api: Any) -> None:
        """Test collecting organization stats for models only."""
        # Mock API responses
        mock_model = Mock()
        mock_model.id = "test/model"
        mock_model.downloads = 100
        mock_model.likes = 50

        mock_model_info = Mock()
        mock_model_info.downloads = 100
        mock_model_info.likes = 50

        mock_model_info_all_time = Mock()
        mock_model_info_all_time.downloads_all_time = 1000

        mock_api_instance = Mock()
        mock_api_instance.list_models.return_value = [mock_model]
        mock_api_instance.model_info.side_effect = [
            mock_model_info,
            mock_model_info_all_time,
        ]
        mock_api.return_value = mock_api_instance

        collector = HFStatsCollector()
        collector.api = mock_api_instance

        results = collector.collect_organization_stats(
            "test-org",
            include_models=True,
            include_datasets=False,
            include_spaces=False,
            max_workers=1,
        )

        assert "models" in results
        assert "datasets" not in results
        assert "spaces" not in results
        assert not results["models"].empty

    def test_collect_organization_stats_no_data(self) -> None:
        """Test collecting organization stats with no data."""
        with patch.object(self.collector, "get_organization_models", return_value=[]):
            with patch.object(
                self.collector, "get_organization_datasets", return_value=[]
            ):
                with patch.object(
                    self.collector, "get_organization_spaces", return_value=[]
                ):
                    results: Dict[str, pd.DataFrame] = (
                        self.collector.collect_organization_stats("test-org")
                    )

                    assert "models" in results
                    assert "datasets" in results
                    assert "spaces" in results
                    assert results["models"].empty
                    assert results["datasets"].empty
                    assert results["spaces"].empty

    @patch("hf_org_stats.HfApi")
    def test_get_detailed_dataset_stats_success(self, mock_api: Any) -> None:
        """Test successful detailed dataset stats collection."""
        # Mock API responses
        mock_dataset_info = Mock()
        mock_dataset_info.downloads = 200
        mock_dataset_info.likes = 75

        mock_dataset_info_all_time = Mock()
        mock_dataset_info_all_time.downloads_all_time = 2000

        mock_api_instance = Mock()
        mock_api_instance.dataset_info.side_effect = [
            mock_dataset_info,
            mock_dataset_info_all_time,
        ]
        mock_api.return_value = mock_api_instance

        collector = HFStatsCollector()
        collector.api = mock_api_instance

        result = collector.get_detailed_dataset_stats("test/dataset")

        assert result["dataset_id"] == "test/dataset"
        assert result["downloads_all_time"] == 2000
        assert result["downloads_30d"] == 200
        assert result["likes"] == 75

    @patch("hf_org_stats.HfApi")
    def test_get_detailed_dataset_stats_exception(self, mock_api: Any) -> None:
        """Test detailed dataset stats with API exception."""
        mock_api_instance = Mock()
        mock_api_instance.dataset_info.side_effect = Exception("API Error")
        mock_api.return_value = mock_api_instance

        collector = HFStatsCollector()
        collector.api = mock_api_instance

        result = collector.get_detailed_dataset_stats("test/dataset")

        assert result["dataset_id"] == "test/dataset"
        assert result["downloads_all_time"] == 0
        assert result["downloads_30d"] == 0
        assert result["likes"] == 0
        assert "error" in result

    @patch("hf_org_stats.HfApi")
    def test_get_detailed_space_stats_success(self, mock_api: Any) -> None:
        """Test successful detailed space stats collection."""
        # Mock API response
        mock_space_info = Mock()
        mock_space_info.likes = 25

        mock_api_instance = Mock()
        mock_api_instance.space_info.return_value = mock_space_info
        mock_api.return_value = mock_api_instance

        collector = HFStatsCollector()
        collector.api = mock_api_instance

        result = collector.get_detailed_space_stats("test/space")

        assert result["space_id"] == "test/space"
        assert result["likes"] == 25

    @patch("hf_org_stats.HfApi")
    def test_get_detailed_space_stats_exception(self, mock_api: Any) -> None:
        """Test detailed space stats with API exception."""
        mock_api_instance = Mock()
        mock_api_instance.space_info.side_effect = Exception("API Error")
        mock_api.return_value = mock_api_instance

        collector = HFStatsCollector()
        collector.api = mock_api_instance

        result = collector.get_detailed_space_stats("test/space")

        assert result["space_id"] == "test/space"
        assert result["likes"] == 0
        assert "error" in result

    @patch("hf_org_stats.HfApi")
    def test_collect_organization_stats_datasets_only(self, mock_api: Any) -> None:
        """Test collecting organization stats for datasets only."""
        # Mock API responses
        mock_dataset = Mock()
        mock_dataset.id = "test/dataset"
        mock_dataset.downloads = 200
        mock_dataset.likes = 75

        mock_dataset_info = Mock()
        mock_dataset_info.downloads = 200
        mock_dataset_info.likes = 75

        mock_dataset_info_all_time = Mock()
        mock_dataset_info_all_time.downloads_all_time = 2000

        mock_api_instance = Mock()
        mock_api_instance.list_datasets.return_value = [mock_dataset]
        mock_api_instance.dataset_info.side_effect = [
            mock_dataset_info,
            mock_dataset_info_all_time,
        ]
        mock_api.return_value = mock_api_instance

        collector = HFStatsCollector()
        collector.api = mock_api_instance

        results = collector.collect_organization_stats(
            "test-org",
            include_models=False,
            include_datasets=True,
            include_spaces=False,
            max_workers=1,
        )

        assert "models" not in results
        assert "datasets" in results
        assert "spaces" not in results
        assert not results["datasets"].empty

    @patch("hf_org_stats.HfApi")
    def test_collect_organization_stats_spaces_only(self, mock_api: Any) -> None:
        """Test collecting organization stats for spaces only."""
        # Mock API responses
        mock_space = Mock()
        mock_space.id = "test/space"
        mock_space.likes = 25

        mock_space_info = Mock()
        mock_space_info.likes = 25

        mock_api_instance = Mock()
        mock_api_instance.list_spaces.return_value = [mock_space]
        mock_api_instance.space_info.return_value = mock_space_info
        mock_api.return_value = mock_api_instance

        collector = HFStatsCollector()
        collector.api = mock_api_instance

        results = collector.collect_organization_stats(
            "test-org",
            include_models=False,
            include_datasets=False,
            include_spaces=True,
            max_workers=1,
        )

        assert "models" not in results
        assert "datasets" not in results
        assert "spaces" in results
        assert not results["spaces"].empty

    @patch("sys.argv", ["hf_org_stats.py", "--organization", "test-org", "--verbose"])
    @patch("hf_org_stats.HFStatsCollector")
    @patch("hf_org_stats.logging")
    def test_main_verbose(self, mock_logging: Any, mock_collector_class: Any) -> None:
        """Test main function with verbose flag."""
        mock_collector = Mock()
        mock_collector.collect_organization_stats.return_value = {
            "models": pd.DataFrame({"model_id": ["test/model"]}),
            "datasets": pd.DataFrame(),
            "spaces": pd.DataFrame(),
        }
        mock_collector.generate_summary.return_value = "Test summary"
        mock_collector.save_results.return_value = ["test.csv"]
        mock_collector_class.return_value = mock_collector

        with patch("hf_org_stats.print") as mock_print:
            main()

            # Verify logging was set up
            mock_logging.getLogger.assert_called()
            mock_logging.getLogger.return_value.setLevel.assert_called_with(
                mock_logging.DEBUG
            )

            # Verify collector was called
            mock_collector.collect_organization_stats.assert_called_once()
            mock_collector.generate_summary.assert_called_once()
            mock_collector.save_results.assert_called_once()

            # Verify output was printed
            mock_print.assert_called()

    @patch("sys.argv", ["hf_org_stats.py", "--organization", "test-org", "--no-save"])
    @patch("hf_org_stats.HFStatsCollector")
    def test_main_no_save(self, mock_collector_class: Any) -> None:
        """Test main function with no-save flag."""
        mock_collector = Mock()
        mock_collector.collect_organization_stats.return_value = {
            "models": pd.DataFrame({"model_id": ["test/model"]}),
            "datasets": pd.DataFrame(),
            "spaces": pd.DataFrame(),
        }
        mock_collector.generate_summary.return_value = "Test summary"
        mock_collector_class.return_value = mock_collector

        with patch("hf_org_stats.print") as mock_print:
            main()

            # Verify collector was called
            mock_collector.collect_organization_stats.assert_called_once()
            mock_collector.generate_summary.assert_called_once()

            # Verify save was NOT called
            mock_collector.save_results.assert_not_called()

            # Verify output was printed
            mock_print.assert_called()

    @patch(
        "sys.argv", ["hf_org_stats.py", "--organization", "test-org", "--models-only"]
    )
    @patch("hf_org_stats.HFStatsCollector")
    def test_main_models_only(self, mock_collector_class: Any) -> None:
        """Test main function with models-only flag."""
        mock_collector = Mock()
        mock_collector.collect_organization_stats.return_value = {
            "models": pd.DataFrame({"model_id": ["test/model"]}),
            "datasets": pd.DataFrame(),
            "spaces": pd.DataFrame(),
        }
        mock_collector.generate_summary.return_value = "Test summary"
        mock_collector.save_results.return_value = ["test.csv"]
        mock_collector_class.return_value = mock_collector

        with patch("hf_org_stats.print") as mock_print:
            main()

            # Verify collector was called with correct parameters
            mock_collector.collect_organization_stats.assert_called_once()
            call_args = mock_collector.collect_organization_stats.call_args
            assert call_args[1]["include_models"] is True
            assert call_args[1]["include_datasets"] is False
            assert call_args[1]["include_spaces"] is False

            # Verify other methods were called
            mock_collector.generate_summary.assert_called_once()
            mock_collector.save_results.assert_called_once()

            # Verify output was printed
            mock_print.assert_called()

    @patch(
        "sys.argv", ["hf_org_stats.py", "--organization", "test-org", "--datasets-only"]
    )
    @patch("hf_org_stats.HFStatsCollector")
    def test_main_datasets_only(self, mock_collector_class: Any) -> None:
        """Test main function with datasets-only flag."""
        mock_collector = Mock()
        mock_collector.collect_organization_stats.return_value = {
            "models": pd.DataFrame(),
            "datasets": pd.DataFrame({"dataset_id": ["test/dataset"]}),
            "spaces": pd.DataFrame(),
        }
        mock_collector.generate_summary.return_value = "Test summary"
        mock_collector.save_results.return_value = ["test.csv"]
        mock_collector_class.return_value = mock_collector

        with patch("hf_org_stats.print") as mock_print:
            main()

            # Verify collector was called with correct parameters
            mock_collector.collect_organization_stats.assert_called_once()
            call_args = mock_collector.collect_organization_stats.call_args
            assert call_args[1]["include_models"] is False
            assert call_args[1]["include_datasets"] is True
            assert call_args[1]["include_spaces"] is False

            # Verify other methods were called
            mock_collector.generate_summary.assert_called_once()
            mock_collector.save_results.assert_called_once()

            # Verify output was printed
            mock_print.assert_called()

    @patch(
        "sys.argv", ["hf_org_stats.py", "--organization", "test-org", "--spaces-only"]
    )
    @patch("hf_org_stats.HFStatsCollector")
    def test_main_spaces_only(self, mock_collector_class: Any) -> None:
        """Test main function with spaces-only flag."""
        mock_collector = Mock()
        mock_collector.collect_organization_stats.return_value = {
            "models": pd.DataFrame(),
            "datasets": pd.DataFrame(),
            "spaces": pd.DataFrame({"space_id": ["test/space"]}),
        }
        mock_collector.generate_summary.return_value = "Test summary"
        mock_collector.save_results.return_value = ["test.csv"]
        mock_collector_class.return_value = mock_collector

        with patch("hf_org_stats.print") as mock_print:
            main()

            # Verify collector was called with correct parameters
            mock_collector.collect_organization_stats.assert_called_once()
            call_args = mock_collector.collect_organization_stats.call_args
            assert call_args[1]["include_models"] is False
            assert call_args[1]["include_datasets"] is False
            assert call_args[1]["include_spaces"] is True

            # Verify other methods were called
            mock_collector.generate_summary.assert_called_once()
            mock_collector.save_results.assert_called_once()

            # Verify output was printed
            mock_print.assert_called()

    @patch(
        "sys.argv",
        ["hf_org_stats.py", "--organization", "test-org", "--output", "json"],
    )
    @patch("hf_org_stats.HFStatsCollector")
    def test_main_output_json(self, mock_collector_class: Any) -> None:
        """Test main function with JSON output format."""
        mock_collector = Mock()
        mock_collector.collect_organization_stats.return_value = {
            "models": pd.DataFrame({"model_id": ["test/model"]}),
            "datasets": pd.DataFrame(),
            "spaces": pd.DataFrame(),
        }
        mock_collector.generate_summary.return_value = "Test summary"
        mock_collector.save_results.return_value = ["test.json"]
        mock_collector_class.return_value = mock_collector

        with patch("hf_org_stats.print") as mock_print:
            main()

            # Verify collector was called
            mock_collector.collect_organization_stats.assert_called_once()
            mock_collector.generate_summary.assert_called_once()

            # Verify save was called with JSON format
            mock_collector.save_results.assert_called_once()
            call_args = mock_collector.save_results.call_args
            assert call_args[1]["output_format"] == "json"

            # Verify output was printed
            mock_print.assert_called()

    @patch(
        "sys.argv",
        ["hf_org_stats.py", "--organization", "test-org", "--output", "excel"],
    )
    @patch("hf_org_stats.HFStatsCollector")
    def test_main_output_excel(self, mock_collector_class: Any) -> None:
        """Test main function with Excel output format."""
        mock_collector = Mock()
        mock_collector.collect_organization_stats.return_value = {
            "models": pd.DataFrame({"model_id": ["test/model"]}),
            "datasets": pd.DataFrame(),
            "spaces": pd.DataFrame(),
        }
        mock_collector.generate_summary.return_value = "Test summary"
        mock_collector.save_results.return_value = ["test.xlsx"]
        mock_collector_class.return_value = mock_collector

        with patch("hf_org_stats.print") as mock_print:
            main()

            # Verify collector was called
            mock_collector.collect_organization_stats.assert_called_once()
            mock_collector.generate_summary.assert_called_once()

            # Verify save was called with Excel format
            mock_collector.save_results.assert_called_once()
            call_args = mock_collector.save_results.call_args
            assert call_args[1]["output_format"] == "excel"

            # Verify output was printed
            mock_print.assert_called()

    @patch(
        "sys.argv",
        ["hf_org_stats.py", "--organization", "test-org", "--max-workers", "10"],
    )
    @patch("hf_org_stats.HFStatsCollector")
    def test_main_max_workers(self, mock_collector_class: Any) -> None:
        """Test main function with custom max workers."""
        mock_collector = Mock()
        mock_collector.collect_organization_stats.return_value = {
            "models": pd.DataFrame({"model_id": ["test/model"]}),
            "datasets": pd.DataFrame(),
            "spaces": pd.DataFrame(),
        }
        mock_collector.generate_summary.return_value = "Test summary"
        mock_collector.save_results.return_value = ["test.csv"]
        mock_collector_class.return_value = mock_collector

        with patch("hf_org_stats.print") as mock_print:
            main()

            # Verify collector was called with correct max workers
            mock_collector.collect_organization_stats.assert_called_once()
            call_args = mock_collector.collect_organization_stats.call_args
            assert call_args[1]["max_workers"] == 10

            # Verify other methods were called
            mock_collector.generate_summary.assert_called_once()
            mock_collector.save_results.assert_called_once()

            # Verify output was printed
            mock_print.assert_called()

    @patch(
        "sys.argv",
        ["hf_org_stats.py", "--organization", "test-org", "--token", "test_token"],
    )
    @patch("hf_org_stats.HFStatsCollector")
    def test_main_with_token(self, mock_collector_class: Any) -> None:
        """Test main function with API token."""
        mock_collector = Mock()
        mock_collector.collect_organization_stats.return_value = {
            "models": pd.DataFrame({"model_id": ["test/model"]}),
            "datasets": pd.DataFrame(),
            "spaces": pd.DataFrame(),
        }
        mock_collector.generate_summary.return_value = "Test summary"
        mock_collector.save_results.return_value = ["test.csv"]
        mock_collector_class.return_value = mock_collector

        with patch("hf_org_stats.print") as mock_print:
            main()

            # Verify collector was initialized with token
            mock_collector_class.assert_called_once_with(token="test_token")

            # Verify other methods were called
            mock_collector.collect_organization_stats.assert_called_once()
            mock_collector.generate_summary.assert_called_once()
            mock_collector.save_results.assert_called_once()

            # Verify output was printed
            mock_print.assert_called()

    @patch("sys.argv", ["hf_org_stats.py", "--organization", "test-org"])
    @patch("hf_org_stats.HFStatsCollector")
    @patch("os.getenv", return_value="env_token")
    def test_main_with_env_token(
        self, mock_getenv: Any, mock_collector_class: Any
    ) -> None:
        """Test main function with environment token."""
        mock_collector = Mock()
        mock_collector.collect_organization_stats.return_value = {
            "models": pd.DataFrame({"model_id": ["test/model"]}),
            "datasets": pd.DataFrame(),
            "spaces": pd.DataFrame(),
        }
        mock_collector.generate_summary.return_value = "Test summary"
        mock_collector.save_results.return_value = ["test.csv"]
        mock_collector_class.return_value = mock_collector

        with patch("hf_org_stats.print") as mock_print:
            main()

            # Verify environment token was checked
            mock_getenv.assert_called_once_with("HF_TOKEN")

            # Verify collector was initialized with environment token
            mock_collector_class.assert_called_once_with(token="env_token")

            # Verify other methods were called
            mock_collector.collect_organization_stats.assert_called_once()
            mock_collector.generate_summary.assert_called_once()
            mock_collector.save_results.assert_called_once()

            # Verify output was printed
            mock_print.assert_called()

    @patch("sys.argv", ["hf_org_stats.py", "--organization", "test-org"])
    @patch("hf_org_stats.HFStatsCollector")
    def test_main_no_data_collected(self, mock_collector_class: Any) -> None:
        """Test main function when no data is collected."""
        mock_collector = Mock()
        mock_collector.collect_organization_stats.return_value = {
            "models": pd.DataFrame(),
            "datasets": pd.DataFrame(),
            "spaces": pd.DataFrame(),
        }
        mock_collector_class.return_value = mock_collector

        with patch("hf_org_stats.print") as mock_print:
            with patch("hf_org_stats.logger") as mock_logger:
                main()

                # Verify collector was called
                mock_collector.collect_organization_stats.assert_called_once()

                # Verify warning was logged
                mock_logger.warning.assert_called_once_with(
                    "No data collected. Exiting."
                )

                # Verify no other methods were called
                mock_collector.generate_summary.assert_not_called()
                mock_collector.save_results.assert_not_called()

                # Verify no output was printed
                mock_print.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
