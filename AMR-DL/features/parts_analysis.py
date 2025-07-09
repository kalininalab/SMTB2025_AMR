#!/usr/bin/env python3
"""
Script to analyze sequence lengths in filtered FASTA files.
Calculates maximum and mean sequence lengths across all files.
"""

import argparse
import glob
import os
import statistics
from typing import Dict, List, Tuple


def parse_fasta_file(file_path: str) -> Dict[str, str]:
    """
    Parse a FASTA file and return a dictionary of sequences.

    Args:
        file_path: Path to the FASTA file

    Returns:
        Dictionary with headers as keys and sequences as values
    """
    sequences = {}
    current_header = None
    current_sequence = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    # Save previous sequence if exists
                    if current_header:
                        sequences[current_header] = "".join(current_sequence)

                    # Start new sequence
                    current_header = line
                    current_sequence = []
                else:
                    # Add to current sequence
                    current_sequence.append(line)

            # Don't forget the last sequence
            if current_header:
                sequences[current_header] = "".join(current_sequence)

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}

    return sequences


def analyze_sequence_lengths(directory: str) -> Tuple[int, float, int, List[int]]:
    """
    Analyze sequence lengths in all FASTA files in the directory.

    Args:
        directory: Directory containing FASTA files

    Returns:
        Tuple of (max_length, mean_length, total_sequences, all_lengths)
    """
    all_lengths = []
    total_files = 0
    total_sequences = 0

    # Get all .ffn files in the directory
    pattern = os.path.join(directory, "*.ffn")
    fasta_files = glob.glob(pattern)

    print(f"Found {len(fasta_files)} FASTA files to analyze")

    for file_path in fasta_files:
        filename = os.path.basename(file_path)
        sequences = parse_fasta_file(file_path)

        if sequences:
            total_files += 1
            file_lengths = [len(seq) for seq in sequences.values()]
            all_lengths.extend(file_lengths)
            total_sequences += len(sequences)

            print(
                f"{filename}: {len(sequences)} sequences, "
                f"lengths: {min(file_lengths)}-{max(file_lengths)}"
            )

    if not all_lengths:
        print("No sequences found in any files!")
        return 0, 0.0, 0, []

    max_length = max(all_lengths)
    mean_length = statistics.mean(all_lengths)

    return max_length, mean_length, total_sequences, all_lengths


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze sequence lengths in FASTA files."
    )
    parser.add_argument("directory", type=str, help="Directory containing FASTA files")
    args = parser.parse_args()

    directory = args.directory

    print("=" * 60)
    print("FASTA Sequence Length Analysis")
    print("=" * 60)

    max_length, mean_length, total_sequences, all_lengths = analyze_sequence_lengths(
        directory
    )

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total sequences analyzed: {total_sequences:,}")
    print(f"Maximum sequence length: {max_length:,} bp")
    print(f"Mean sequence length: {mean_length:.2f} bp")

    if all_lengths:
        print(f"Minimum sequence length: {min(all_lengths):,} bp")
        print(f"Median sequence length: {statistics.median(all_lengths):.2f} bp")
        print(f"Standard deviation: {statistics.stdev(all_lengths):.2f} bp")

        # Show distribution
        print("\nLength distribution:")
        length_ranges = [
            (0, 500),
            (500, 1000),
            (1000, 2000),
            (2000, 5000),
            (5000, float("inf")),
        ]
        for min_len, max_len in length_ranges:
            if max_len == float("inf"):
                count = sum(1 for length in all_lengths if length >= min_len)
                print(
                    f"  {min_len:,}+ bp: {count:,} sequences ({count / len(all_lengths) * 100:.1f}%)"
                )
            else:
                count = sum(1 for length in all_lengths if min_len <= length < max_len)
                print(
                    f"  {min_len:,}-{max_len:,} bp: {count:,} sequences ({count / len(all_lengths) * 100:.1f}%)"
                )


if __name__ == "__main__":
    main()
