#!/usr/bin/env python3
"""
FFN to CSV Converter

This script converts FFN (FASTA nucleotide) files to CSV format.
It processes all .ffn files in a given directory and creates corresponding .csv files.

Usage:
    python ffn_to_csv.py <directory_path>

Example:
    python ffn_to_csv.py /path/to/ffn/files/
"""

import argparse
import csv
import logging
import sys
from pathlib import Path


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_ffn_file(ffn_file_path):
    """
    Parse an FFN file and extract sequence information.

    Args:
        ffn_file_path (str): Path to the FFN file

    Returns:
        list: List of dictionaries containing sequence data
    """
    sequences = []
    current_sequence = None
    current_seq_data = ""

    try:
        with open(ffn_file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if line.startswith(">"):
                    # Save previous sequence if exists
                    if current_sequence is not None:
                        sequences.append(
                            {
                                "sequence_id": current_sequence["id"],
                                "description": current_sequence["description"],
                                "sequence": current_seq_data,
                                "sequence_length": len(current_seq_data),
                            }
                        )

                    # Parse header line
                    header_parts = line[1:].split(" ", 1)
                    seq_id = header_parts[0]
                    description = header_parts[1] if len(header_parts) > 1 else ""

                    current_sequence = {"id": seq_id, "description": description}
                    current_seq_data = ""

                else:
                    # Accumulate sequence data
                    current_seq_data += line

            # Add the last sequence
            if current_sequence is not None:
                sequences.append(
                    {
                        "sequence_id": current_sequence["id"],
                        "description": current_sequence["description"],
                        "sequence": current_seq_data,
                        "sequence_length": len(current_seq_data),
                    }
                )

    except IOError as e:
        logging.error("Error parsing %s: %s", ffn_file_path, str(e))
        return []

    return sequences


def write_csv_file(sequences, output_path):
    """
    Write sequence data to a CSV file.

    Args:
        sequences (list): List of sequence dictionaries
        output_path (str): Path to output CSV file
    """
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["sequence_id", "description", "sequence", "sequence_length"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header
            writer.writeheader()

            # Write sequences
            for seq in sequences:
                writer.writerow(seq)

        logging.info(
            "Successfully wrote %d sequences to %s", len(sequences), output_path
        )

    except IOError as e:
        logging.error("Error writing CSV file %s: %s", output_path, str(e))


def convert_ffn_to_csv(ffn_file_path, output_dir=None):
    """
    Convert a single FFN file to CSV format.

    Args:
        ffn_file_path (str): Path to the FFN file
        output_dir (str): Directory for output CSV file (optional)
    """
    ffn_path = Path(ffn_file_path)

    # Determine output path
    if output_dir:
        output_path = Path(output_dir) / f"{ffn_path.stem}.csv"
    else:
        output_path = ffn_path.with_suffix(".csv")

    # Parse FFN file
    logging.info("Processing %s", ffn_file_path)
    sequences = parse_ffn_file(ffn_file_path)

    if not sequences:
        logging.warning("No sequences found in %s", ffn_file_path)
        return

    # Write CSV file
    write_csv_file(sequences, output_path)


def process_directory(directory_path, output_dir=None):
    """
    Process all FFN files in a directory.

    Args:
        directory_path (str): Path to directory containing FFN files
        output_dir (str): Directory for output CSV files (optional)
    """
    dir_path = Path(directory_path)

    if not dir_path.exists():
        logging.error("Directory %s does not exist", directory_path)
        return

    if not dir_path.is_dir():
        logging.error("%s is not a directory", directory_path)
        return

    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Find all FFN files
    ffn_files = list(dir_path.glob("*.ffn"))

    if not ffn_files:
        logging.warning("No FFN files found in %s", directory_path)
        return

    logging.info("Found %d FFN files to process", len(ffn_files))

    # Process each FFN file
    for ffn_file in ffn_files:
        try:
            convert_ffn_to_csv(str(ffn_file), output_dir)
        except IOError as e:
            logging.error("Error processing %s: %s", ffn_file, str(e))


def main():
    """Main function to handle command line arguments and execute conversion"""
    parser = argparse.ArgumentParser(
        description="Convert FFN (FASTA nucleotide) files to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ffn_to_csv.py /path/to/ffn/files/
    python ffn_to_csv.py /path/to/ffn/files/ -o /path/to/output/
    python ffn_to_csv.py /path/to/single/file.ffn
        """,
    )

    parser.add_argument(
        "input_path",
        help="Path to directory containing FFN files or path to single FFN file",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output directory for CSV files (default: same as input directory/file location)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input_path)

    # Check if input is a file or directory
    if input_path.is_file():
        if input_path.suffix.lower() == ".ffn":
            convert_ffn_to_csv(str(input_path), args.output)
        else:
            logging.error("Input file %s is not an FFN file", input_path)
    elif input_path.is_dir():
        process_directory(str(input_path), args.output)
    else:
        logging.error("Input path %s does not exist", input_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
