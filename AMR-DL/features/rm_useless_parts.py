import json
import os
import re
from glob import glob

# TODO: Change this to be commands args
path = (
    "/Users/danielkorkin/Documents/SMTB2025/Lab16/SMTB2025_AMR/AMR-DL/Data/Raw/pa_ffn"
)

# Next for each dictionary only keep the keys and content where the keys have one of the items in the following list in their keys (use Whole Word Match)
keys = [
    # Table 1 gene names
    "dacB",
    "mpl",
    "ampC",
    "sltB1",
    "nalD",
    "wapH",
    "wbpL",
    "ssg",
    "PA4333",
    "PA3172",
    "ilvC",
    "infA",
    "clpA",
    "spoT",
    "rpsA",
    "PA1767",
    "phoQ",
    "pilP",
    "PA2982",
    "vacJ",
    # Table 2 gene names
    "dacB",
    "mpl",
    "ampR",
    "nalD",
    "mexB",
    "nalC",
    "orfN",
    "pgi",
    "clpA",
    "hepP",
    "minC",
    "phoQ",
    "PA14_55770",
    "tatC",
    # Table 3 mutations
    "dacB",
    "mpl",
    "nalC",
    "nalD",
    "ampC",
    "mexR",
    # Table 4 mutated genes
    "dacB",
    "mpl",
    "nalD",
    "ampC",
    "mexR",
    # Table 5 proteins
    "AmpC",
    "AmpR",
    "DacB",
    "MexB",
    "MexR",
    "Mpl",
    "NalC",
    "NalD",
    "PhoQ",
    "SltB1",
]


# 1. Get all fasta files in the directory
def get_fasta_files(directory):
    """Get all FASTA files in the directory."""
    fasta_extensions = ["*.fasta", "*.fa", "*.fna", "*.ffn"]
    fasta_files = []
    for ext in fasta_extensions:
        fasta_files.extend(glob(os.path.join(directory, ext)))
    return fasta_files


# 2. Parse FASTA file into dictionary
def parse_fasta_to_dict(fasta_file):
    """Parse a FASTA file into a dictionary."""
    fasta_dict = {}
    current_header = None
    current_sequence = []

    with open(fasta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # If we have a previous sequence, save it
                if current_header:
                    fasta_dict[current_header] = "".join(current_sequence)

                # Start new sequence
                current_header = line
                current_sequence = []
            else:
                # Add to current sequence
                current_sequence.append(line)

        # Don't forget the last sequence
        if current_header:
            fasta_dict[current_header] = "".join(current_sequence)

    return fasta_dict


# 3. Filter dictionary based on gene keys (whole word match)
def filter_by_gene_keys(fasta_dict, gene_keys):
    """Filter dictionary to keep only entries that contain any of the gene keys as whole words."""
    filtered_results = {}

    for header, sequence in fasta_dict.items():
        # Check if any of the gene keys appear as whole words in the header
        for gene_key in gene_keys:
            # Create regex pattern for whole word match (case-insensitive)
            pattern = r"\b" + re.escape(gene_key) + r"\b"
            if re.search(pattern, header, re.IGNORECASE):
                filtered_results[header] = sequence
                break  # Found a match, no need to check other keys

    return filtered_results


# 4. Process all FASTA files
def process_all_fasta_files(directory, gene_keys):
    """Process all FASTA files in the directory."""
    fasta_files = get_fasta_files(directory)
    print(f"Found {len(fasta_files)} FASTA files in {directory}")

    all_results = {}

    for fasta_file in fasta_files:
        file_name = os.path.basename(fasta_file)
        print(f"Processing {file_name}...")

        # Parse FASTA to dictionary
        fasta_dict = parse_fasta_to_dict(fasta_file)
        print(f"  Found {len(fasta_dict)} sequences")

        # Filter by gene keys
        filtered_results = filter_by_gene_keys(fasta_dict, gene_keys)
        print(f"  Filtered to {len(filtered_results)} sequences")

        # Store results
        all_results[file_name] = filtered_results

        # Print some examples if found
        if filtered_results:
            print("  Sample matches:")
            for header in list(filtered_results.keys())[:3]:
                print(f"    {header}")

    return all_results


# 5. Save filtered sequences to new FFN files
def save_filtered_ffn_files(results, output_directory):
    """Save filtered sequences to new FFN files in the output directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    for original_filename, filtered_sequences in results.items():
        if filtered_sequences:  # Only create files if there are sequences to save
            # Create output filename
            output_filename = f"filtered_{original_filename}"
            output_path = os.path.join(output_directory, output_filename)

            # Write filtered sequences to new FFN file
            with open(output_path, "w", encoding="utf-8") as f:
                for header, sequence in filtered_sequences.items():
                    f.write(f"{header}\n")
                    # Write sequence with line breaks every 60 characters (standard FASTA format)
                    for i in range(0, len(sequence), 60):
                        f.write(f"{sequence[i : i + 60]}\n")

            print(
                f"Created filtered file: {output_path} ({len(filtered_sequences)} sequences)"
            )
        else:
            print(f"No sequences to save for {original_filename}")


# 6. Main execution
if __name__ == "__main__":
    # Remove duplicates from keys list
    unique_keys = list(set(keys))
    print(f"Looking for {len(unique_keys)} unique gene keys:")
    print(unique_keys)
    print()

    # Process all files
    results = process_all_fasta_files(path, unique_keys)

    # Print summary
    print("\nSummary:")
    total_matches = 0
    for file_name, filtered_results in results.items():
        match_count = len(filtered_results)
        total_matches += match_count
        print(f"{file_name}: {match_count} matches")

    print(f"\nTotal matches across all files: {total_matches}")

    # Save results to JSON file
    output_file = os.path.join(path, "filtered_sequences.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    # Save filtered sequences to new FFN files
    output_directory = os.path.join(path, "filtered_ffn_files")
    print(f"\nCreating filtered FFN files in: {output_directory}")
    save_filtered_ffn_files(results, output_directory)
