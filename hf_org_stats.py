#!/usr/bin/env python3
"""
Hugging Face Organization Statistics Collector

A professional tool to collect comprehensive statistics for Hugging Face organizations,
including models, datasets, and spaces with download counts and engagement metrics.

Author: AI Assistant
License: MIT
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HFStatsCollector:
    """
    Professional collector for Hugging Face organization statistics.

    This class provides methods to collect comprehensive statistics for models,
    datasets, and spaces within a Hugging Face organization, including download
    counts (all-time and 30-day) and engagement metrics (likes).

    Attributes:
        api: Hugging Face Hub API client
        session: HTTP session for direct API calls
    """

    def __init__(self, token: Optional[str] = None) -> None:
        """
        Initialize the HFStatsCollector.

        Args:
            token: Hugging Face API token for higher rate limits
        """
        self.api = HfApi(token=token)
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
            logger.info("Initialized with API token for enhanced rate limits")
        else:
            logger.warning("No API token provided - using public rate limits")

    def _get_item_id(self, item: object, id_attrs: List[str]) -> Optional[str]:
        """Get item ID from various possible attribute names."""
        for attr in id_attrs:
            if hasattr(item, attr):
                value = getattr(item, attr)
                if value is not None:
                    return str(value)
        return None

    def _get_safe_attr(self, item: object, attr_name: str, default: int = 0) -> int:
        """Safely get attribute value with fallback to default."""
        try:
            value = getattr(item, attr_name)
            return int(value) if value is not None else default
        except (AttributeError, ValueError, TypeError):
            return default

    def get_organization_models(self, organization: str) -> List[Dict]:
        """
        Fetch all models from an organization with basic statistics.

        Args:
            organization: Organization name (e.g., 'arcee-ai')

        Returns:
            List of model information dictionaries
        """
        logger.info(f"Fetching models for organization: {organization}")

        try:
            models = list(self.api.list_models(author=organization))

            model_list = []
            for model in models:
                model_id = self._get_item_id(model, ["id", "modelId"])
                if model_id:
                    model_info = {
                        "model_id": model_id,
                        "downloads": self._get_safe_attr(model, "downloads"),
                        "likes": self._get_safe_attr(model, "likes"),
                    }
                    model_list.append(model_info)

            logger.info(f"Found {len(model_list)} models for {organization}")
            return model_list

        except Exception as e:
            logger.error(f"Error fetching models for {organization}: {e}")
            return []

    def get_organization_datasets(self, organization: str) -> List[Dict]:
        """
        Fetch all datasets from an organization with basic statistics.

        Args:
            organization: Organization name (e.g., 'arcee-ai')

        Returns:
            List of dataset information dictionaries
        """
        logger.info(f"Fetching datasets for organization: {organization}")

        try:
            datasets = list(self.api.list_datasets(author=organization))

            dataset_list = []
            for dataset in datasets:
                dataset_id = self._get_item_id(dataset, ["id", "datasetId"])
                if dataset_id:
                    dataset_info = {
                        "dataset_id": dataset_id,
                        "downloads": self._get_safe_attr(dataset, "downloads"),
                        "likes": self._get_safe_attr(dataset, "likes"),
                    }
                    dataset_list.append(dataset_info)

            logger.info(f"Found {len(dataset_list)} datasets for {organization}")
            return dataset_list

        except Exception as e:
            logger.error(f"Error fetching datasets for {organization}: {e}")
            return []

    def get_organization_spaces(self, organization: str) -> List[Dict]:
        """
        Fetch all spaces from an organization with basic statistics.

        Args:
            organization: Organization name (e.g., 'arcee-ai')

        Returns:
            List of space information dictionaries
        """
        logger.info(f"Fetching spaces for organization: {organization}")

        try:
            spaces = list(self.api.list_spaces(author=organization))

            space_list = []
            for space in spaces:
                space_id = self._get_item_id(space, ["id", "spaceId"])
                if space_id:
                    space_info = {
                        "space_id": space_id,
                        "likes": self._get_safe_attr(space, "likes"),
                    }
                    space_list.append(space_info)

            logger.info(f"Found {len(space_list)} spaces for {organization}")
            return space_list

        except Exception as e:
            logger.error(f"Error fetching spaces for {organization}: {e}")
            return []

    def get_detailed_model_stats(self, model_id: str) -> Dict:
        """
        Get comprehensive statistics for a specific model.

        Makes two API calls to get both 30-day and all-time download statistics,
        as the Hugging Face API requires separate calls for these metrics.

        Args:
            model_id: Full model ID (e.g., 'arcee-ai/llama-2-7b-hf')

        Returns:
            Dictionary with model statistics
        """
        try:
            # Get 30-day downloads and likes
            model_info = self.api.model_info(model_id)
            downloads_30d = self._get_safe_attr(model_info, "downloads")
            likes = self._get_safe_attr(model_info, "likes")

            # Get all-time downloads
            model_info_all_time = self.api.model_info(
                model_id, expand=["downloadsAllTime"]
            )
            downloads_all_time = self._get_safe_attr(
                model_info_all_time, "downloads_all_time"
            )

            return {
                "model_id": model_id,
                "downloads_all_time": downloads_all_time,
                "downloads_30d": downloads_30d,
                "likes": likes,
            }

        except Exception as e:
            logger.warning(f"Error fetching detailed stats for {model_id}: {e}")
            return {
                "model_id": model_id,
                "downloads_all_time": 0,
                "downloads_30d": 0,
                "likes": 0,
                "error": str(e),
            }

    def get_detailed_dataset_stats(self, dataset_id: str) -> Dict:
        """
        Get comprehensive statistics for a specific dataset.

        Args:
            dataset_id: Full dataset ID (e.g., 'arcee-ai/dataset-name')

        Returns:
            Dictionary with dataset statistics
        """
        try:
            # Get 30-day downloads and likes
            dataset_info = self.api.dataset_info(dataset_id)
            downloads_30d = self._get_safe_attr(dataset_info, "downloads")
            likes = self._get_safe_attr(dataset_info, "likes")

            # Get all-time downloads
            dataset_info_all_time = self.api.dataset_info(
                dataset_id, expand=["downloadsAllTime"]
            )
            downloads_all_time = self._get_safe_attr(
                dataset_info_all_time, "downloads_all_time"
            )

            return {
                "dataset_id": dataset_id,
                "downloads_all_time": downloads_all_time,
                "downloads_30d": downloads_30d,
                "likes": likes,
            }

        except Exception as e:
            logger.warning(f"Error fetching detailed stats for {dataset_id}: {e}")
            return {
                "dataset_id": dataset_id,
                "downloads_all_time": 0,
                "downloads_30d": 0,
                "likes": 0,
                "error": str(e),
            }

    def get_detailed_space_stats(self, space_id: str) -> Dict:
        """
        Get comprehensive statistics for a specific space.

        Args:
            space_id: Full space ID (e.g., 'arcee-ai/space-name')

        Returns:
            Dictionary with space statistics
        """
        try:
            space_info = self.api.space_info(space_id)
            likes = self._get_safe_attr(space_info, "likes")

            return {
                "space_id": space_id,
                "likes": likes,
            }

        except Exception as e:
            logger.warning(f"Error fetching detailed stats for {space_id}: {e}")
            return {
                "space_id": space_id,
                "likes": 0,
                "error": str(e),
            }

    def _collect_detailed_stats_parallel(
        self,
        items: List[Dict],
        id_key: str,
        stats_func: Callable[[str], Dict],
        max_workers: int,
        desc: str,
    ) -> List[Dict]:
        """
        Collect detailed statistics using parallel processing.

        Args:
            items: List of item dictionaries
            id_key: Key for the ID field
            stats_func: Function to get detailed stats
            max_workers: Maximum number of parallel workers
            desc: Description for progress bar

        Returns:
            List of detailed statistics dictionaries
        """
        detailed_stats = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(stats_func, item[id_key]): item[id_key]
                for item in items
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(items), desc=desc, leave=False) as pbar:
                for future in as_completed(future_to_item):
                    item_id = future_to_item[future]
                    try:
                        result = future.result()
                        detailed_stats.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {item_id}: {e}")
                        # Create error result based on item type
                        if id_key == "model_id":
                            detailed_stats.append(
                                {
                                    "model_id": item_id,
                                    "downloads_all_time": 0,
                                    "downloads_30d": 0,
                                    "likes": 0,
                                    "error": str(e),
                                }
                            )
                        elif id_key == "dataset_id":
                            detailed_stats.append(
                                {
                                    "dataset_id": item_id,
                                    "downloads_all_time": 0,
                                    "downloads_30d": 0,
                                    "likes": 0,
                                    "error": str(e),
                                }
                            )
                        elif id_key == "space_id":
                            detailed_stats.append(
                                {
                                    "space_id": item_id,
                                    "likes": 0,
                                    "error": str(e),
                                }
                            )
                    finally:
                        pbar.update(1)

        return detailed_stats

    def collect_organization_stats(
        self,
        organization: str,
        include_models: bool = True,
        include_datasets: bool = True,
        include_spaces: bool = True,
        max_workers: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect comprehensive statistics for an organization.

        Args:
            organization: Organization name
            include_models: Whether to collect model statistics
            include_datasets: Whether to collect dataset statistics
            include_spaces: Whether to collect space statistics
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary with DataFrames for each type
        """
        logger.info(f"Starting comprehensive data collection for {organization}")

        results = {}

        # Collect models
        if include_models:
            models = self.get_organization_models(organization)
            if models:
                logger.info(
                    f"Collecting detailed statistics for {len(models)} models "
                    f"using {max_workers} parallel workers"
                )
                detailed_stats = self._collect_detailed_stats_parallel(
                    models,
                    "model_id",
                    self.get_detailed_model_stats,
                    max_workers,
                    "Fetching model stats",
                )
                df = pd.DataFrame(detailed_stats)
                df.fillna(0, inplace=True)
                results["models"] = df
                logger.info("Completed model statistics collection")
            else:
                results["models"] = pd.DataFrame()

        # Collect datasets
        if include_datasets:
            datasets = self.get_organization_datasets(organization)
            if datasets:
                logger.info(
                    f"Collecting detailed statistics for {len(datasets)} datasets "
                    f"using {max_workers} parallel workers"
                )
                detailed_stats = self._collect_detailed_stats_parallel(
                    datasets,
                    "dataset_id",
                    self.get_detailed_dataset_stats,
                    max_workers,
                    "Fetching dataset stats",
                )
                df = pd.DataFrame(detailed_stats)
                df.fillna(0, inplace=True)
                results["datasets"] = df
                logger.info("Completed dataset statistics collection")
            else:
                results["datasets"] = pd.DataFrame()

        # Collect spaces
        if include_spaces:
            spaces = self.get_organization_spaces(organization)
            if spaces:
                logger.info(
                    f"Collecting detailed statistics for {len(spaces)} spaces "
                    f"using {max_workers} parallel workers"
                )
                detailed_stats = self._collect_detailed_stats_parallel(
                    spaces,
                    "space_id",
                    self.get_detailed_space_stats,
                    max_workers,
                    "Fetching space stats",
                )
                df = pd.DataFrame(detailed_stats)
                df.fillna(0, inplace=True)
                results["spaces"] = df
                logger.info("Completed space statistics collection")
            else:
                results["spaces"] = pd.DataFrame()

        return results

    def save_results(
        self,
        results: Dict[str, pd.DataFrame],
        organization: str,
        output_format: str = "csv",
    ) -> List[str]:
        """
        Save results to files in the specified format.

        Args:
            results: Dictionary with DataFrames for each type
            organization: Organization name
            output_format: Output format ('csv', 'json', 'excel')

        Returns:
            List of saved filenames
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []

        for item_type, df in results.items():
            if df.empty:
                continue

            if output_format == "csv":
                filename = f"{organization}_{item_type}_stats_{timestamp}.csv"
                df.to_csv(filename, index=False)
            elif output_format == "json":
                filename = f"{organization}_{item_type}_stats_{timestamp}.json"
                df.to_json(filename, orient="records", indent=2)
            elif output_format == "excel":
                filename = f"{organization}_{item_type}_stats_{timestamp}.xlsx"
                df.to_excel(filename, index=False)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            logger.info(f"{item_type.capitalize()} results saved to: {filename}")
            saved_files.append(filename)

        return saved_files

    def generate_summary(
        self, results: Dict[str, pd.DataFrame], organization: str
    ) -> str:
        """
        Generate a comprehensive summary report.

        Args:
            results: Dictionary with DataFrames for each type
            organization: Organization name

        Returns:
            Formatted summary string
        """
        summary_lines = [f"=== Summary for {organization} ==="]

        # Models summary
        if "models" in results and not results["models"].empty:
            df = results["models"]
            summary_lines.extend(
                [
                    f"\n--- Models ({len(df)} total) ---",
                    f"Total all-time downloads: {df['downloads_all_time'].sum():,}",
                    f"Average all-time downloads per model: "
                    f"{df['downloads_all_time'].mean():,.0f}",
                    f"Most downloaded model (all-time): "
                    f"{df['downloads_all_time'].max():,} downloads",
                    f"Total 30-day downloads: {df['downloads_30d'].sum():,}",
                    f"Average 30-day downloads per model: "
                    f"{df['downloads_30d'].mean():,.0f}",
                    f"Most downloaded model (30-day): "
                    f"{df['downloads_30d'].max():,} downloads",
                    f"Total likes: {df['likes'].sum():,}",
                    f"Average likes per model: {df['likes'].mean():,.0f}",
                    f"Most liked model: {df['likes'].max():,} likes",
                ]
            )

            # Top 5 models by all-time downloads
            summary_lines.append("\nTop 5 models by all-time downloads:")
            top_models = df.nlargest(5, "downloads_all_time")[
                ["model_id", "downloads_all_time", "downloads_30d", "likes"]
            ]
            for _, row in top_models.iterrows():
                summary_lines.append(
                    f"  {row['model_id']}: {row['downloads_all_time']:,} all-time, "
                    f"{row['downloads_30d']:,} 30d, {row['likes']:,} likes"
                )

        # Datasets summary
        if "datasets" in results and not results["datasets"].empty:
            df = results["datasets"]
            summary_lines.extend(
                [
                    f"\n--- Datasets ({len(df)} total) ---",
                    f"Total all-time downloads: {df['downloads_all_time'].sum():,}",
                    f"Average all-time downloads per dataset: "
                    f"{df['downloads_all_time'].mean():,.0f}",
                    f"Most downloaded dataset (all-time): "
                    f"{df['downloads_all_time'].max():,} downloads",
                    f"Total 30-day downloads: {df['downloads_30d'].sum():,}",
                    f"Average 30-day downloads per dataset: "
                    f"{df['downloads_30d'].mean():,.0f}",
                    f"Most downloaded dataset (30-day): "
                    f"{df['downloads_30d'].max():,} downloads",
                    f"Total likes: {df['likes'].sum():,}",
                    f"Average likes per dataset: {df['likes'].mean():,.0f}",
                    f"Most liked dataset: {df['likes'].max():,} likes",
                ]
            )

            # Top 5 datasets by all-time downloads
            summary_lines.append("\nTop 5 datasets by all-time downloads:")
            top_datasets = df.nlargest(5, "downloads_all_time")[
                ["dataset_id", "downloads_all_time", "downloads_30d", "likes"]
            ]
            for _, row in top_datasets.iterrows():
                summary_lines.append(
                    f"  {row['dataset_id']}: {row['downloads_all_time']:,} all-time, "
                    f"{row['downloads_30d']:,} 30d, {row['likes']:,} likes"
                )

        # Spaces summary
        if "spaces" in results and not results["spaces"].empty:
            df = results["spaces"]
            summary_lines.extend(
                [
                    f"\n--- Spaces ({len(df)} total) ---",
                    f"Total likes: {df['likes'].sum():,}",
                    f"Average likes per space: {df['likes'].mean():,.0f}",
                    f"Most liked space: {df['likes'].max():,} likes",
                ]
            )

            # Top 5 spaces by likes
            summary_lines.append("\nTop 5 spaces by likes:")
            top_spaces = df.nlargest(5, "likes")[["space_id", "likes"]]
            for _, row in top_spaces.iterrows():
                summary_lines.append(f"  {row['space_id']}: {row['likes']:,} likes")

        return "\n".join(summary_lines)


def main() -> None:
    """Main function to run the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Professional Hugging Face Organization Statistics Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            """
Examples:
  python hf_org_stats.py --organization arcee-ai
  python hf_org_stats.py --organization microsoft --models-only --output excel
  python hf_org_stats.py --organization openai --max-workers 10 --token YOUR_TOKEN
            """
        ),
    )

    parser.add_argument(
        "--organization",
        "-o",
        default="arcee-ai",
        help="Organization name (default: arcee-ai)",
    )
    parser.add_argument(
        "--token", "-t", help="Hugging Face API token for higher rate limits"
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=5,
        help="Maximum number of parallel workers (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-f",
        choices=["csv", "json", "excel"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save results to file"
    )
    parser.add_argument(
        "--models-only", action="store_true", help="Only collect model statistics"
    )
    parser.add_argument(
        "--datasets-only", action="store_true", help="Only collect dataset statistics"
    )
    parser.add_argument(
        "--spaces-only", action="store_true", help="Only collect space statistics"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine what to collect
    if args.models_only:
        include_models, include_datasets, include_spaces = True, False, False
    elif args.datasets_only:
        include_models, include_datasets, include_spaces = False, True, False
    elif args.spaces_only:
        include_models, include_datasets, include_spaces = False, False, True
    else:
        include_models, include_datasets, include_spaces = True, True, True

    # Use token from environment if not provided
    token = args.token or os.getenv("HF_TOKEN")

    # Initialize collector
    collector = HFStatsCollector(token=token)

    # Collect statistics
    results = collector.collect_organization_stats(
        args.organization,
        include_models=include_models,
        include_datasets=include_datasets,
        include_spaces=include_spaces,
        max_workers=args.max_workers,
    )

    # Check if any data was collected
    if not any(not df.empty for df in results.values()):
        logger.warning("No data collected. Exiting.")
        return

    # Generate and print summary
    summary = collector.generate_summary(results, args.organization)
    print(summary)

    # Save results
    if not args.no_save:
        saved_files = collector.save_results(results, args.organization, args.output)
        print(f"\nResults saved to {len(saved_files)} file(s)")


if __name__ == "__main__":
    main()
