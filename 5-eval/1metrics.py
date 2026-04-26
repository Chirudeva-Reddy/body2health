#!/usr/bin/env python3
"""
Comprehensive training metrics extraction script for the BodyFit project.

This script parses training logs and dataset statistics to generate structured
data files ready for visualization and analysis.

Usage:
    python 5-eval/1metrics.py

Output files:
    - data/training_metrics.csv: Epoch-by-epoch training metrics
    - data/training_metrics.json: Complete training metrics in JSON format
    - data/dataset_statistics.csv: Dataset statistics summary
    - data/training_summary.json: Training summary and best performance
    - data/loss_components.csv: Loss component breakdown over time
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Optional, Tuple, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.eval.training_logs import (  # noqa: E402
    parse_batch_log as shared_parse_batch_log,
    parse_training_log as shared_parse_training_log,
)

# Create data directory for logging
os.makedirs('data', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TrainingMetricsExtractor:
    """Extract and process training metrics from log files."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.epoch_metrics = []
        self.batch_metrics = []
        self.best_model_epochs = []
        self.final_metrics = {}
        
        # Best-model regex stays here; the shared parser only handles the
        # epoch / batch progress lines.
        self.best_model_pattern = re.compile(
            r'New best model saved to (checkpoints/[\w_./]+)'
        )

    def parse_training_log(self) -> Dict[str, Any]:
        """Parse the training log file and extract all metrics."""
        logger.info(f"Parsing training log: {self.log_path}")

        if not os.path.exists(self.log_path):
            raise FileNotFoundError(f"Training log not found: {self.log_path}")

        # Epoch and batch rows come from the shared parser to keep the regex
        # for the standard log format in one place.
        epoch_rows = shared_parse_training_log(self.log_path)
        batch_rows = shared_parse_batch_log(self.log_path)
        for r in batch_rows:
            self.batch_metrics.append({
                'epoch': r.epoch,
                'batch': r.batch,
                'total_batches': r.total_batches,
                'total_loss': r.loss,
                'contrastive_loss': r.contrastive,
                'regression_loss': r.regression,
            })
        for r in epoch_rows:
            self.epoch_metrics.append({
                'epoch': r.epoch,
                'train_loss': r.train_loss,
                'val_loss': r.val_loss,
                'bmi_mae': r.val_mae_bmi,
                'body_fat_mae': r.val_mae_bf,
            })

        # Best-model lines must be associated with the epoch they appear after,
        # so we still scan the file once for them and use the running tail of
        # epoch_metrics as the active epoch.
        with open(self.log_path, 'r', encoding='utf-8') as f:
            running_idx = 0
            for line in f:
                line = line.strip()
                # Advance running_idx whenever we cross another epoch summary.
                if 'Train Loss' in line and 'Val Loss' in line:
                    running_idx = min(running_idx + 1, len(self.epoch_metrics))
                best_match = self.best_model_pattern.search(line)
                if best_match and running_idx > 0:
                    checkpoint_path = best_match.group(1)
                    epoch_metric = self.epoch_metrics[running_idx - 1]
                    self.best_model_epochs.append({
                        'epoch': epoch_metric['epoch'],
                        'checkpoint_path': checkpoint_path,
                        'bmi_mae': epoch_metric['bmi_mae'],
                        'body_fat_mae': epoch_metric['body_fat_mae'],
                        'val_loss': epoch_metric['val_loss']
                    })
        
        logger.info(f"Parsed {len(self.epoch_metrics)} epochs and {len(self.batch_metrics)} batches")
        logger.info(f"Found {len(self.best_model_epochs)} best model checkpoints")
        
        return self._compile_metrics()
    
    def _compile_metrics(self) -> Dict[str, Any]:
        """Compile all extracted metrics into a structured format."""
        if not self.epoch_metrics:
            raise ValueError("No epoch metrics found in log file")
        
        # Calculate additional metrics
        for i, epoch in enumerate(self.epoch_metrics):
            # Add convergence metrics
            if i > 0:
                prev_epoch = self.epoch_metrics[i-1]
                epoch['train_loss_delta'] = epoch['train_loss'] - prev_epoch['train_loss']
                epoch['val_loss_delta'] = epoch['val_loss'] - prev_epoch['val_loss']
                epoch['bmi_mae_delta'] = epoch['bmi_mae'] - prev_epoch['bmi_mae']
                epoch['body_fat_mae_delta'] = epoch['body_fat_mae'] - prev_epoch['body_fat_mae']
            else:
                epoch['train_loss_delta'] = 0.0
                epoch['val_loss_delta'] = 0.0
                epoch['bmi_mae_delta'] = 0.0
                epoch['body_fat_mae_delta'] = 0.0
            
            # Add moving averages (window=5)
            window = 5
            start_idx = max(0, i - window + 1)
            window_epochs = self.epoch_metrics[start_idx:i+1]
            epoch['train_loss_ma5'] = mean([e['train_loss'] for e in window_epochs])
            epoch['val_loss_ma5'] = mean([e['val_loss'] for e in window_epochs])
            epoch['bmi_mae_ma5'] = mean([e['bmi_mae'] for e in window_epochs])
            epoch['body_fat_mae_ma5'] = mean([e['body_fat_mae'] for e in window_epochs])
        
        # Identify best performing epochs
        best_bmi_epoch = min(self.epoch_metrics, key=lambda x: x['bmi_mae'])
        best_bf_epoch = min(self.epoch_metrics, key=lambda x: x['body_fat_mae'])
        best_val_loss_epoch = min(self.epoch_metrics, key=lambda x: x['val_loss'])
        
        # Compile final metrics
        final_epoch = self.epoch_metrics[-1]
        self.final_metrics = {
            'total_epochs': len(self.epoch_metrics),
            'final_epoch': final_epoch['epoch'],
            'final_train_loss': final_epoch['train_loss'],
            'final_val_loss': final_epoch['val_loss'],
            'final_bmi_mae': final_epoch['bmi_mae'],
            'final_body_fat_mae': final_epoch['body_fat_mae'],
            'best_bmi_epoch': {
                'epoch': best_bmi_epoch['epoch'],
                'bmi_mae': best_bmi_epoch['bmi_mae'],
                'body_fat_mae': best_bmi_epoch['body_fat_mae'],
                'val_loss': best_bmi_epoch['val_loss']
            },
            'best_body_fat_epoch': {
                'epoch': best_bf_epoch['epoch'],
                'bmi_mae': best_bf_epoch['bmi_mae'],
                'body_fat_mae': best_bf_epoch['body_fat_mae'],
                'val_loss': best_bf_epoch['val_loss']
            },
            'best_val_loss_epoch': {
                'epoch': best_val_loss_epoch['epoch'],
                'bmi_mae': best_val_loss_epoch['bmi_mae'],
                'body_fat_mae': best_val_loss_epoch['body_fat_mae'],
                'val_loss': best_val_loss_epoch['val_loss']
            },
            'training_summary': {
                'train_loss_improvement': self.epoch_metrics[0]['train_loss'] - final_epoch['train_loss'],
                'val_loss_improvement': self.epoch_metrics[0]['val_loss'] - final_epoch['val_loss'],
                'bmi_mae_improvement': self.epoch_metrics[0]['bmi_mae'] - final_epoch['bmi_mae'],
                'body_fat_mae_improvement': self.epoch_metrics[0]['body_fat_mae'] - final_epoch['body_fat_mae'],
            }
        }
        
        return {
            'epochs': self.epoch_metrics,
            'batches': self.batch_metrics,
            'best_models': self.best_model_epochs,
            'final_metrics': self.final_metrics,
            'extraction_info': {
                'log_file': self.log_path,
                'extraction_timestamp': datetime.now().isoformat(),
                'total_lines_processed': len(self.batch_metrics) + len(self.epoch_metrics) + len(self.best_model_epochs)
            }
        }
    
    def export_to_csv(self, output_dir: str = "data"):
        """Export metrics to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export epoch metrics
        epoch_csv_path = os.path.join(output_dir, "training_metrics.csv")
        if self.epoch_metrics:
            with open(epoch_csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = self.epoch_metrics[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.epoch_metrics)
            logger.info(f"Exported epoch metrics to: {epoch_csv_path}")
        
        # Export batch metrics (sample every 10th batch to reduce size)
        batch_csv_path = os.path.join(output_dir, "batch_metrics_sample.csv")
        if self.batch_metrics:
            sample_batches = self.batch_metrics[::10]  # Every 10th batch
            with open(batch_csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = sample_batches[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sample_batches)
            logger.info(f"Exported batch metrics sample to: {batch_csv_path}")
        
        # Export best model checkpoints
        best_models_csv_path = os.path.join(output_dir, "best_models.csv")
        if self.best_model_epochs:
            with open(best_models_csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = self.best_model_epochs[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.best_model_epochs)
            logger.info(f"Exported best models to: {best_models_csv_path}")
        
        # Export loss components breakdown
        self._export_loss_components(output_dir)
    
    def _export_loss_components(self, output_dir: str):
        """Export loss component analysis."""
        if not self.batch_metrics:
            return
        
        # Group batches by epoch and calculate loss component statistics
        epoch_loss_stats = []
        for epoch in range(1, len(self.epoch_metrics) + 1):
            epoch_batches = [b for b in self.batch_metrics if b['epoch'] == epoch]
            if not epoch_batches:
                continue
            
            contrastive_losses = [b['contrastive_loss'] for b in epoch_batches]
            regression_losses = [b['regression_loss'] for b in epoch_batches]
            
            stats = {
                'epoch': epoch,
                'contrastive_loss_mean': mean(contrastive_losses),
                'contrastive_loss_std': pstdev(contrastive_losses) if len(contrastive_losses) > 1 else 0.0,
                'contrastive_loss_min': min(contrastive_losses),
                'contrastive_loss_max': max(contrastive_losses),
                'regression_loss_mean': mean(regression_losses),
                'regression_loss_std': pstdev(regression_losses) if len(regression_losses) > 1 else 0.0,
                'regression_loss_min': min(regression_losses),
                'regression_loss_max': max(regression_losses),
                'total_batches': len(epoch_batches)
            }
            epoch_loss_stats.append(stats)
        
        if epoch_loss_stats:
            loss_csv_path = os.path.join(output_dir, "loss_components.csv")
            with open(loss_csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = epoch_loss_stats[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(epoch_loss_stats)
            logger.info(f"Exported loss components to: {loss_csv_path}")
    
    def export_to_json(self, output_dir: str = "data"):
        """Export all metrics to JSON format."""
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, "training_metrics.json")
        metrics_data = self._compile_metrics()
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        logger.info(f"Exported complete metrics to: {json_path}")
        
        # Export summary
        summary_path = os.path.join(output_dir, "training_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.final_metrics, f, indent=2, default=str)
        logger.info(f"Exported training summary to: {summary_path}")


class DatasetStatisticsAnalyzer:
    """Analyze dataset statistics using the existing 3stats.py script."""
    
    def __init__(self, labels_csv: str, pairs_csv: str):
        self.labels_csv = labels_csv
        self.pairs_csv = pairs_csv
        
    def run_analysis(self) -> Dict[str, Any]:
        """Run dataset statistics analysis."""
        logger.info("Running dataset statistics analysis...")
        
        # Check if CSV files exist
        if not os.path.exists(self.labels_csv):
            logger.warning(f"Labels CSV not found: {self.labels_csv}")
            return {}
        
        if not os.path.exists(self.pairs_csv):
            logger.warning(f"Pairs CSV not found: {self.pairs_csv}")
            return {}
        
        try:
            # Run the existing 3stats.py script
            result = subprocess.run([
                sys.executable, "1-data/3stats.py",
                "--labels", self.labels_csv,
                "--pairs", self.pairs_csv,
                "--out", "data/dataset_stats_temp.md"
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                logger.error(f"Dataset stats script failed: {result.stderr}")
                return {}
            
            # Parse the markdown output
            stats_data = self._parse_dataset_stats_output()
            return stats_data
            
        except Exception as e:
            logger.error(f"Error running dataset analysis: {e}")
            return {}
    
    def _parse_dataset_stats_output(self) -> Dict[str, Any]:
        """Parse the markdown output from 3stats.py."""
        stats_file = "data/dataset_stats_temp.md"
        if not os.path.exists(stats_file):
            return {}
        
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract statistics from markdown format
            stats_data = {}
            lines = content.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('## '):
                    current_section = line[3:].strip()
                    stats_data[current_section] = {}
                elif line.startswith('- ') and current_section:
                    # Parse line like: "- height_cm: n=1234, min=150.00, max=200.00, mean=175.50, median=175.00, std=12.34"
                    stat_match = re.match(r'- (\w+): n=(\d+), min=([\d.]+), max=([\d.]+), mean=([\d.]+), median=([\d.]+), std=([\d.]+)', line)
                    if stat_match:
                        metric, count, min_val, max_val, mean_val, median_val, std_val = stat_match.groups()
                        stats_data[current_section][metric] = {
                            'count': int(count),
                            'min': float(min_val),
                            'max': float(max_val),
                            'mean': float(mean_val),
                            'median': float(median_val),
                            'std': float(std_val)
                        }
            
            return stats_data
            
        except Exception as e:
            logger.error(f"Error parsing dataset stats: {e}")
            return {}
        finally:
            # Clean up temporary file
            if os.path.exists(stats_file):
                os.remove(stats_file)
    
    def export_dataset_stats_csv(self, stats_data: Dict[str, Any], output_dir: str = "data"):
        """Export dataset statistics to CSV format."""
        if not stats_data:
            logger.warning("No dataset statistics to export")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, "dataset_statistics.csv")
        
        # Flatten the nested dictionary structure
        flat_data = []
        for section, metrics in stats_data.items():
            for metric, stats in metrics.items():
                row = {
                    'section': section,
                    'metric': metric,
                    **stats
                }
                flat_data.append(row)
        
        if flat_data:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = flat_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat_data)
            logger.info(f"Exported dataset statistics to: {csv_path}")


def main():
    """Main function to run the training metrics extraction."""
    parser = argparse.ArgumentParser(description="Extract training metrics from logs and analyze dataset statistics")
    parser.add_argument("--log-file", type=str, default="out/train_640x480.log", help="Path to training log file")
    parser.add_argument("--labels-csv", type=str, default="data/labels.csv", help="Path to labels CSV file")
    parser.add_argument("--pairs-csv", type=str, default="data/pairs.csv", help="Path to pairs CSV file")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for extracted data")
    args = parser.parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Extract training metrics
        logger.info("=" * 60)
        logger.info("TRAINING METRICS EXTRACTION")
        logger.info("=" * 60)
        
        extractor = TrainingMetricsExtractor(args.log_file)
        metrics_data = extractor.parse_training_log()
        
        # Export training metrics
        extractor.export_to_csv(args.output_dir)
        extractor.export_to_json(args.output_dir)
        
        # Analyze dataset statistics
        logger.info("=" * 60)
        logger.info("DATASET STATISTICS ANALYSIS")
        logger.info("=" * 60)
        
        dataset_analyzer = DatasetStatisticsAnalyzer(args.labels_csv, args.pairs_csv)
        dataset_stats = dataset_analyzer.run_analysis()
        dataset_analyzer.export_dataset_stats_csv(dataset_stats, args.output_dir)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 60)
        
        final_metrics = extractor.final_metrics
        logger.info(f"Total epochs processed: {final_metrics['total_epochs']}")
        logger.info(f"Final BMI MAE: {final_metrics['final_bmi_mae']:.3f}")
        logger.info(f"Final Body Fat MAE: {final_metrics['final_body_fat_mae']:.3f}%")
        logger.info(f"Best BMI MAE: {final_metrics['best_bmi_epoch']['bmi_mae']:.3f} (Epoch {final_metrics['best_bmi_epoch']['epoch']})")
        logger.info(f"Best Body Fat MAE: {final_metrics['best_body_fat_epoch']['body_fat_mae']:.3f}% (Epoch {final_metrics['best_body_fat_epoch']['epoch']})")
        logger.info(f"Best validation loss: {final_metrics['best_val_loss_epoch']['val_loss']:.3f} (Epoch {final_metrics['best_val_loss_epoch']['epoch']})")
        
        logger.info(f"\nOutput files created in '{args.output_dir}/':")
        output_files = [
            "training_metrics.csv",
            "training_metrics.json", 
            "training_summary.json",
            "batch_metrics_sample.csv",
            "best_models.csv",
            "loss_components.csv",
            "dataset_statistics.csv"
        ]
        
        for filename in output_files:
            filepath = os.path.join(args.output_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"  ✓ {filename}")
            else:
                logger.info(f"  ✗ {filename} (not created)")
        
        logger.info("\nExtraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
