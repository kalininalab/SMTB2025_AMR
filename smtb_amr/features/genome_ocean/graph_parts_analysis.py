#!/usr/bin/env python3
"""
Script to generate histograms of sequence lengths from CSV files in a directory.
Usage: python graph_parts_analysis.py <directory_path>
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_csv_files(directory):
    """
    Read all CSV files in the specified directory and extract sequence lengths.

    Args:
        directory (str): Path to directory containing CSV files

    Returns:
        dict: Dictionary with filename as key and list of sequence lengths as value
    """
    csv_data = {}
    directory_path = Path(directory)

    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return csv_data

    csv_files = list(directory_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in directory '{directory}'")
        return csv_data

    for csv_file in csv_files:
        try:
            sequence_lengths = []
            with open(csv_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                # Check if the required column exists
                if "sequence_length" not in reader.fieldnames:
                    print(
                        f"Warning: 'sequence_length' column not found in {csv_file.name}"
                    )
                    continue

                # Extract sequence lengths
                for row in reader:
                    length_str = row["sequence_length"].strip()
                    if length_str:  # Skip empty values
                        try:
                            length = int(length_str)
                            sequence_lengths.append(length)
                        except ValueError:
                            # Skip non-numeric values
                            continue

            if sequence_lengths:
                csv_data[csv_file.name] = sequence_lengths
                print(f"Loaded {len(sequence_lengths)} sequences from {csv_file.name}")
            else:
                print(f"Warning: No valid sequence lengths found in {csv_file.name}")

        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")

    return csv_data


def create_histograms(csv_data):
    """
    Create histograms for sequence lengths from all CSV files.

    Args:
        csv_data (dict): Dictionary with filename as key and sequence lengths as value

    Returns:
        plotly.graph_objects.Figure: The created figure object
    """
    if not csv_data:
        print("No data to plot.")
        return None

    # Calculate number of subplots needed
    num_files = len(csv_data)
    cols = min(5, num_files)  # Maximum 5 columns for better layout
    rows = (num_files + cols - 1) // cols  # Ceiling division

    print(f"Creating plot with {rows} rows and {cols} columns for {num_files} files")

    # Create subplot titles with statistics
    subplot_titles = []
    for filename, sequence_lengths in csv_data.items():
        n = len(sequence_lengths)
        std_length = np.std(sequence_lengths)
        subplot_titles.append(f"{filename}<br>(n={n}, std={std_length:.1f})")

    # Calculate appropriate spacing based on number of rows
    # For very large numbers of subplots, use minimal spacing
    if rows > 50:
        vertical_spacing = 0.005  # Very small spacing for many rows
        horizontal_spacing = 0.005
    else:
        vertical_spacing = max(0.02, min(0.08, 1.0 / (rows + 1)))
        horizontal_spacing = max(0.02, min(0.08, 1.0 / (cols + 1)))

    # Create figure with subplots
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )

    # Create histogram for each file
    for i, (filename, sequence_lengths) in enumerate(csv_data.items()):
        row = (i // cols) + 1
        col = (i % cols) + 1

        # Calculate statistics
        mean_length = np.mean(sequence_lengths)
        median_length = np.median(sequence_lengths)

        # Calculate bin edges with 10 bp bins
        min_length = min(sequence_lengths)
        max_length = max(sequence_lengths)
        bin_start = (min_length // 10) * 10  # Round down to nearest 10
        bin_end = ((max_length // 10) + 1) * 10  # Round up to nearest 10

        # Create histogram trace
        fig.add_trace(
            go.Histogram(
                x=sequence_lengths,
                xbins=dict(
                    start=bin_start,
                    end=bin_end,
                    size=10,  # 10 bp bins
                ),
                opacity=0.7,
                name=f"{filename}",
                showlegend=False,
                marker_color="skyblue",
                marker_line_color="black",
                marker_line_width=0.5,
            ),
            row=row,
            col=col,
        )

        # Add vertical lines for mean and median
        fig.add_vline(
            x=mean_length,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Mean: {mean_length:.0f}",
            annotation_position="top right",
            row=row,
            col=col,
        )

        fig.add_vline(
            x=median_length,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"Median: {median_length:.0f}",
            annotation_position="top left",
            row=row,
            col=col,
        )

        # Update subplot axes with 100 bp tick intervals
        fig.update_xaxes(
            title_text="Sequence Length (bp)",
            dtick=100,  # 100 bp intervals on x-axis
            row=row,
            col=col,
        )
        fig.update_yaxes(title_text="Frequency", row=row, col=col)

    # Update layout
    fig.update_layout(
        title_text="Sequence Length Distributions",
        title_x=0.5,
        height=max(2000, 200 * rows),  # Increased minimum height for many subplots
        width=max(1600, 320 * cols),  # Adjusted width
        showlegend=False,
    )

    return fig


def main():
    """
    Main function to parse arguments and create histograms.
    """
    parser = argparse.ArgumentParser(
        description="Create histograms of sequence lengths from CSV files"
    )
    parser.add_argument("directory", help="Directory containing CSV files")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file to save the plot (default: sequence_length_histograms.png)",
        default="sequence_length_histograms.png",
    )
    parser.add_argument(
        "--format",
        "-f",
        help="Output format (png, html, pdf, svg)",
        default="png",
        choices=["png", "html", "pdf", "svg"],
    )

    args = parser.parse_args()

    # Read CSV files and extract sequence lengths
    print(f"Reading CSV files from: {args.directory}")
    csv_data = read_csv_files(args.directory)

    if not csv_data:
        print("No valid CSV files with sequence_length column found.")
        return

    # Create histograms
    print("Creating histograms...")
    fig = create_histograms(csv_data)

    if fig is None:
        print("Failed to create histograms.")
        return

    # Save plot to file
    try:
        print(f"Saving plot to: {args.output}")

        if args.format == "html":
            fig.write_html(args.output)
        elif args.format == "pdf":
            fig.write_image(args.output, format="pdf", width=2000, height=3000)
        elif args.format == "svg":
            fig.write_image(args.output, format="svg", width=2000, height=3000)
        else:  # png
            fig.write_image(args.output, format="png", width=2000, height=3000, scale=1)

        print(f"Plot successfully saved to: {args.output}")

    except Exception as e:
        print(f"Error saving plot: {e}")
        print(
            "Note: For image formats (png, pdf, svg), you may need to install: pip install kaleido"
        )


if __name__ == "__main__":
    main()
