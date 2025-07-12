#!/usr/bin/env python3
"""
sorting.py - Organize embeddings data by bacteria type using genotype files

This script reads embeddings data and organizes it by bacteria type using
genotype files that contain sample IDs for each bacteria species.
"""

import pandas as pd
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_genotype_mappings(genotype_dir):
    """
    Load sample ID to bacteria type mappings from genotype files.
    
    Args:
        genotype_dir (str): Path to directory containing genotype files
        
    Returns:
        dict: Mapping of sample_id -> bacteria_type
    """
    genotype_mappings = {}
    genotype_path = Path(genotype_dir)
    
    if not genotype_path.exists():
        raise FileNotFoundError(f"Genotype directory not found: {genotype_dir}")
    
    # Find all genotype files
    genotype_files = list(genotype_path.glob("*_all_genotypes.txt"))
    
    if not genotype_files:
        raise FileNotFoundError(f"No genotype files found in {genotype_dir}")
    
    logger.info(f"Found {len(genotype_files)} genotype files")
    
    for genotype_file in genotype_files:
        # Extract bacteria name from filename
        bacteria_name = genotype_file.stem.replace("_all_genotypes", "")
        logger.info(f"Processing {bacteria_name}")
        
        # Read sample IDs from file
        try:
            with open(genotype_file, 'r') as f:
                sample_ids = [line.strip() for line in f if line.strip()]
            
            # Map each sample ID to bacteria type
            for sample_id in sample_ids:
                if sample_id in genotype_mappings:
                    logger.warning(f"Duplicate sample ID found: {sample_id}")
                genotype_mappings[sample_id] = bacteria_name
            
            logger.info(f"Loaded {len(sample_ids)} samples for {bacteria_name}")
            
        except Exception as e:
            logger.error(f"Error reading {genotype_file}: {e}")
            continue
    
    logger.info(f"Total samples mapped: {len(genotype_mappings)}")
    return genotype_mappings


def load_embeddings(embeddings_file):
    """
    Load embeddings data from TSV file.
    
    Args:
        embeddings_file (str): Path to embeddings TSV file
        
    Returns:
        pd.DataFrame: Embeddings data with sample_id as index
    """
    logger.info(f"Loading embeddings from {embeddings_file}")
    
    try:
        # Read the TSV file
        embeddings_df = pd.read_csv(embeddings_file, sep='\t')
        
        # Set sample_id as index if it's a column
        if 'sample_id' in embeddings_df.columns:
            embeddings_df.set_index('sample_id', inplace=True)
        
        logger.info(f"Loaded embeddings for {len(embeddings_df)} samples with {len(embeddings_df.columns)} features")
        return embeddings_df
        
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        raise


def sort_embeddings_by_bacteria(embeddings_df, genotype_mappings):
    """
    Sort embeddings by bacteria type using genotype mappings.
    
    Args:
        embeddings_df (pd.DataFrame): Embeddings data
        genotype_mappings (dict): Sample ID to bacteria type mapping
        
    Returns:
        dict: Dictionary of bacteria_type -> DataFrame
    """
    sorted_embeddings = {}
    unmatched_samples = []
    
    # Add bacteria type column to embeddings
    embeddings_df['bacteria_type'] = embeddings_df.index.map(genotype_mappings)
    
    # Identify unmatched samples
    unmatched_mask = embeddings_df['bacteria_type'].isna()
    unmatched_samples = embeddings_df[unmatched_mask].index.tolist()
    
    if unmatched_samples:
        logger.warning(f"Found {len(unmatched_samples)} samples without bacteria type mapping")
        logger.info(f"First few unmatched samples: {unmatched_samples[:5]}")
    
    # Group by bacteria type
    matched_df = embeddings_df.dropna(subset=['bacteria_type'])
    
    for bacteria_type, group_df in matched_df.groupby('bacteria_type'):
        # Remove bacteria_type column for clean embeddings data
        clean_df = group_df.drop('bacteria_type', axis=1)
        sorted_embeddings[bacteria_type] = clean_df
        logger.info(f"{bacteria_type}: {len(clean_df)} samples")
    
    return sorted_embeddings, unmatched_samples


def save_sorted_embeddings(sorted_embeddings, output_dir, format='tsv'):
    """
    Save sorted embeddings to separate files by bacteria type.
    
    Args:
        sorted_embeddings (dict): Dictionary of bacteria_type -> DataFrame
        output_dir (str): Output directory path
        format (str): Output format ('tsv' or 'csv')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving sorted embeddings to {output_dir}")
    
    for bacteria_type, df in sorted_embeddings.items():
        if format == 'tsv':
            filename = f"{bacteria_type}_embeddings.tsv"
            filepath = output_path / filename
            df.to_csv(filepath, sep='\t')
        elif format == 'csv':
            filename = f"{bacteria_type}_embeddings.csv"
            filepath = output_path / filename
            df.to_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {bacteria_type}: {len(df)} samples -> {filename}")


def create_summary_report(sorted_embeddings, unmatched_samples, output_dir):
    """
    Create a summary report of the sorting results.
    
    Args:
        sorted_embeddings (dict): Dictionary of bacteria_type -> DataFrame
        unmatched_samples (list): List of unmatched sample IDs
        output_dir (str): Output directory path
    """
    summary_path = Path(output_dir) / "sorting_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("Embeddings Sorting Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total bacteria types: {len(sorted_embeddings)}\n")
        total_matched = sum(len(df) for df in sorted_embeddings.values())
        f.write(f"Total matched samples: {total_matched}\n")
        f.write(f"Total unmatched samples: {len(unmatched_samples)}\n\n")
        
        f.write("Breakdown by bacteria type:\n")
        f.write("-" * 30 + "\n")
        for bacteria_type, df in sorted_embeddings.items():
            f.write(f"{bacteria_type}: {len(df)} samples\n")
        
        if unmatched_samples:
            f.write("\nFirst 20 unmatched samples:\n")
            f.write("-" * 30 + "\n")
            for sample in unmatched_samples[:20]:
                f.write(f"{sample}\n")
    
    logger.info(f"Summary report saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Sort embeddings by bacteria type using genotype files")
    parser.add_argument(
        "--embeddings", 
        default="../../combined_averaged_embeddings.tsv",
        help="Path to embeddings TSV file (default: ../../combined_averaged_embeddings.tsv)"
    )
    parser.add_argument(
        "--genotypes", 
        default="../../Full_Genotype",
        help="Path to genotype directory (default: ../../Full_Genotype)"
    )
    parser.add_argument(
        "--output", 
        default="./sorted_embeddings",
        help="Output directory for sorted embeddings (default: ./sorted_embeddings)"
    )
    parser.add_argument(
        "--format", 
        choices=['tsv', 'csv'], 
        default='tsv',
        help="Output format (default: tsv)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load genotype mappings
        logger.info("Step 1: Loading genotype mappings...")
        genotype_mappings = load_genotype_mappings(args.genotypes)
        
        # Load embeddings
        logger.info("Step 2: Loading embeddings...")
        embeddings_df = load_embeddings(args.embeddings)
        
        # Sort embeddings by bacteria type
        logger.info("Step 3: Sorting embeddings by bacteria type...")
        sorted_embeddings, unmatched_samples = sort_embeddings_by_bacteria(embeddings_df, genotype_mappings)
        
        # Save sorted embeddings
        logger.info("Step 4: Saving sorted embeddings...")
        save_sorted_embeddings(sorted_embeddings, args.output, args.format)
        
        # Create summary report
        logger.info("Step 5: Creating summary report...")
        create_summary_report(sorted_embeddings, unmatched_samples, args.output)
        
        logger.info("Sorting completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during sorting: {e}")
        raise


if __name__ == "__main__":
    main()
