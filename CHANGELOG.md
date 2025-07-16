# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup
- Professional README with badges and emojis
- Comprehensive .gitignore file
- pyproject.toml for modern Python packaging
- Contributing guidelines
- Changelog tracking

### Changed
- Enhanced README formatting and structure
- Added professional badges for Python version, license, and dependencies

## [1.0.0] - 2024-12-01

### Added
- Initial release of Hugging Face Organization Statistics tool
- Command-line interface for collecting organization statistics
- Support for models, datasets, and spaces
- Parallel processing with configurable workers
- Multiple output formats (CSV, JSON, Excel)
- Comprehensive statistics including download counts and engagement metrics
- API token support for higher rate limits
- Progress tracking with tqdm
- Detailed summary reports
- Error handling and logging

### Features
- **Models**: Collect download counts (all-time and 30-day) and likes
- **Datasets**: Collect download counts (all-time and 30-day) and likes
- **Spaces**: Collect likes and engagement metrics
- **Parallel Processing**: Fast data collection with configurable workers
- **Multiple Output Formats**: CSV, JSON, and Excel
- **Command Line Interface**: Full-featured CLI with comprehensive options

### Technical Details
- Python 3.7+ compatibility
- Uses Hugging Face Hub API
- Pandas for data processing
- ThreadPoolExecutor for parallel processing
- Environment variable support for API tokens
- Comprehensive error handling and logging
