#!/usr/bin/env python3
"""
Script to extract and combine genotypes from Splits folder.
Reads test_validation.txt, test.txt, train.txt, validation.txt from each bacteria folder
and combines them into a single file per bacteria.
"""

import argparse
from pathlib import Path
from typing import Set


def read_genotype_file(file_path: Path) -> Set[str]:
    """
    Read genotype strains from a text file.
    
    Args:
        file_path: Path to the genotype file
        
    Returns:
        Set of strain IDs
    """
    strains = set()
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    strain = line.strip()
                    if strain:  # Skip empty lines
                        strains.add(strain)
        except (IOError, OSError) as e:
            print(f"Warning: Could not read {file_path}: {e}")
    else:
        print(f"Warning: File not found: {file_path}")
    
    return strains


def get_bacteria_genotypes(bacteria_dir: Path) -> Set[str]:
    """
    Get all genotypes for a specific bacteria by combining all split files.
    
    Args:
        bacteria_dir: Path to the bacteria directory in Splits folder
        
    Returns:
        Set of all strain IDs for this bacteria
    """
    split_files = ['test_validation.txt', 'test.txt', 'train.txt', 'validation.txt']
    all_strains = set()
    
    for split_file in split_files:
        file_path = bacteria_dir / split_file
        strains = read_genotype_file(file_path)
        all_strains.update(strains)
        print(f"  {split_file}: {len(strains)} strains")
    
    return all_strains


def process_splits_folder(input_dir: Path, output_dir: Path):
    """
    Process all bacteria in the Splits folder and generate combined genotype files.
    
    Args:
        input_dir: Path to the Splits folder
        output_dir: Path to the output directory
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all bacteria directories
    bacteria_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not bacteria_dirs:
        print(f"No bacteria directories found in {input_dir}")
        return
    
    bacteria_dirs.sort()  # Sort for consistent processing order
    
    print(f"Found {len(bacteria_dirs)} bacteria directories:")
    for bacteria_dir in bacteria_dirs:
        print(f"  {bacteria_dir.name}")
    
    print("\nProcessing bacteria genotypes...")
    
    # Process each bacteria
    for bacteria_dir in bacteria_dirs:
        bacteria_name = bacteria_dir.name
        print(f"\nProcessing {bacteria_name}:")
        
        # Get all genotypes for this bacteria
        all_strains = get_bacteria_genotypes(bacteria_dir)
        
        if all_strains:
            # Sort strains for consistent output
            sorted_strains = sorted(list(all_strains))
            
            # Write to output file
            output_file = output_dir / f"{bacteria_name}_all_genotypes.txt"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for strain in sorted_strains:
                        f.write(f"{strain}\n")
                
                print(f"  Total unique strains: {len(sorted_strains)}")
                print(f"  Output written to: {output_file}")
            except (IOError, OSError) as e:
                print(f"  Error writing to {output_file}: {e}")
        else:
            print(f"  No strains found for {bacteria_name}")


def main():
    """Main function to handle CLI arguments and orchestrate the process."""
    parser = argparse.ArgumentParser(
        description="Extract and combine genotypes from Splits folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_genotype.py /path/to/AMR-DL/Data/Splits /path/to/output
  python get_genotype.py ../../Data/Splits ./genotype_outputs
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to the Splits folder containing bacteria directories'
    )
    
    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to the output directory where combined genotype files will be saved'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects and validate
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return 1
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process the splits folder
    try:
        process_splits_folder(input_dir, output_dir)
        print("\nProcessing completed successfully!")
        print(f"Combined genotype files saved to: {output_dir}")
    except (IOError, OSError) as e:
        print(f"Error during processing: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())