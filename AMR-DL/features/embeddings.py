#!/usr/bin/env python3
"""
FFN to Embeddings Processing Script

This script processes FFN (FASTA Nucleotide) files and generates embeddings using GenomeOcean-4B model.
Features:
1. Processes all FFN files from a specified directory
2. Groups sequences in batches of 3 for efficient processing
3. Uses caching to avoid reprocessing identical sequences
4. Outputs embeddings as TSV files

Usage:
    python embeddings.py [input_directory] [output_directory]
"""

import hashlib
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GenomeOceanEmbedder:
    """Wrapper class for GenomeOcean model operations"""

    def __init__(
        self, model_name: str = "pGenomeOcean/GenomeOcean-4B", device: str = "auto"
    ):
        """
        Initialize the GenomeOcean model and tokenizer

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._get_device(device)

        logger.info(f"Loading GenomeOcean model: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Log GPU information if available
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.1f} GB")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )

        # Load model directly on target device
        if self.device == "cuda":
            try:
                # Try with accelerate (device_map)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
                logger.info(
                    "Successfully loaded model with Flash Attention 2 and device_map"
                )
            except ValueError as e:
                if "accelerate" in str(e):
                    logger.warning(
                        "accelerate not available, loading without device_map"
                    )
                    try:
                        # Fallback: Load without device_map
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.bfloat16,
                            attn_implementation="flash_attention_2",
                        ).to(self.device)
                        logger.info("Successfully loaded model with Flash Attention 2")
                    except Exception as e2:
                        logger.warning(
                            f"Flash Attention 2 failed, falling back to standard attention: {e2}"
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.bfloat16,
                        ).to(self.device)
                        logger.info("Successfully loaded model with standard attention")
                else:
                    raise e
            except Exception as e:
                logger.warning(
                    f"Flash Attention 2 failed, falling back to standard attention: {e}"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                ).to(self.device)
                logger.info("Successfully loaded model with standard attention")
        else:
            # For CPU, don't use flash attention
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use float32 for CPU
            ).to(self.device)

        logger.info("Model loaded successfully")

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                # Set GPU memory management for better performance
                torch.cuda.empty_cache()
                return "cuda"
            else:
                logger.warning("CUDA not available, falling back to CPU")
                return "cpu"
        return device

    def generate_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a batch of sequences

        Args:
            sequences: List of DNA sequences

        Returns:
            Tensor of embeddings with shape (batch_size, embedding_dim)
        """
        try:
            # Clear GPU cache before processing if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Tokenize sequences
            output = self.tokenizer.batch_encode_plus(
                sequences,
                max_length=10240,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )

            input_ids = output["input_ids"].to(self.device)
            attention_mask = output["attention_mask"].to(self.device)

            # Generate embeddings
            with torch.no_grad():
                model_output = (
                    self.model.forward(
                        input_ids=input_ids, attention_mask=attention_mask
                    )[0]
                    .detach()
                    .cpu()
                )

            attention_mask_cpu = attention_mask.unsqueeze(-1).detach().cpu()

            # Apply attention masking and average pooling
            embedding = torch.sum(model_output * attention_mask_cpu, dim=1) / torch.sum(
                attention_mask_cpu, dim=1
            )

            # Clear variables from GPU memory
            del input_ids, attention_mask, model_output, attention_mask_cpu
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return embedding

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Clear GPU memory on error
            if self.device == "cuda":
                torch.cuda.empty_cache()
            raise


class EmbeddingCache:
    """Simple file-based caching system for embeddings"""

    def __init__(self, cache_dir: str):
        """
        Initialize cache

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "sequence_embeddings.pkl"

        # Load existing cache
        self._cache = self._load_cache()
        logger.info(f"Cache initialized with {len(self._cache)} entries")

    def _load_cache(self) -> Dict[str, np.ndarray]:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            logger.error(f"Could not save cache: {e}")

    def _get_sequence_hash(self, sequence: str) -> str:
        """Generate hash for a sequence"""
        return hashlib.md5(sequence.encode()).hexdigest()

    def get_embedding(self, sequence: str) -> Optional[np.ndarray]:
        """Get embedding from cache if it exists"""
        seq_hash = self._get_sequence_hash(sequence)
        return self._cache.get(seq_hash)

    def store_embedding(self, sequence: str, embedding: np.ndarray):
        """Store embedding in cache"""
        seq_hash = self._get_sequence_hash(sequence)
        self._cache[seq_hash] = embedding

    def save(self):
        """Save cache to disk"""
        self._save_cache()


def read_ffn_file(file_path: str) -> List[Tuple[str, str]]:
    """
    Read sequences from an FFN file

    Args:
        file_path: Path to the FFN file

    Returns:
        List of (header, sequence) tuples
    """
    sequences = []
    try:
        with open(file_path, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequences.append((record.description, str(record.seq)))
        logger.info(f"Read {len(sequences)} sequences from {file_path}")
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

    return sequences


def process_sequences_in_batches(
    sequences: List[Tuple[str, str]],
    embedder: GenomeOceanEmbedder,
    cache: EmbeddingCache,
    batch_size: int = 3,
) -> List[Tuple[str, np.ndarray]]:
    """
    Process sequences in batches and generate embeddings

    Args:
        sequences: List of (header, sequence) tuples
        embedder: GenomeOcean embedder instance
        cache: Embedding cache instance
        batch_size: Number of sequences to process at once

    Returns:
        List of (header, embedding) tuples
    """
    results = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]

        # Check cache for existing embeddings
        cached_embeddings = []
        uncached_sequences = []
        uncached_headers = []

        for j, (header, seq) in enumerate(batch):
            cached_emb = cache.get_embedding(seq)
            if cached_emb is not None:
                cached_embeddings.append((header, cached_emb))
                logger.debug(f"Using cached embedding for sequence {i + j + 1}")
            else:
                uncached_sequences.append(seq)
                uncached_headers.append(header)

        # Process uncached sequences
        if uncached_sequences:
            logger.info(
                f"Processing batch {i // batch_size + 1}: {len(uncached_sequences)} new sequences"
            )

            # Log GPU memory usage if available
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                gpu_memory_cached = torch.cuda.memory_reserved() / 1e9
                logger.debug(
                    f"GPU Memory - Used: {gpu_memory_used:.2f} GB, Cached: {gpu_memory_cached:.2f} GB"
                )

            try:
                embeddings = embedder.generate_embeddings(uncached_sequences)

                # Store in cache and results
                for k, (header, embedding) in enumerate(
                    zip(uncached_headers, embeddings)
                ):
                    emb_numpy = embedding.numpy()
                    cache.store_embedding(uncached_sequences[k], emb_numpy)
                    results.append((header, emb_numpy))

            except Exception as e:
                logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                continue

        # Add cached embeddings to results
        results.extend(cached_embeddings)

    return results


def save_embeddings_to_tsv(embeddings: List[Tuple[str, np.ndarray]], output_file: str):
    """
    Save embeddings to a TSV file

    Args:
        embeddings: List of (header, embedding) tuples
        output_file: Output TSV file path
    """
    try:
        # Create DataFrame
        data = []
        for header, embedding in embeddings:
            row = {"sequence_header": header}
            # Add embedding dimensions as columns
            for i, val in enumerate(embedding):
                row[f"emb_{i}"] = val
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(output_file, sep="\t", index=False)
        logger.info(f"Saved {len(embeddings)} embeddings to {output_file}")

    except Exception as e:
        logger.error(f"Error saving embeddings to {output_file}: {e}")


def process_ffn_directory(input_dir: str, output_dir: str, cache_dir: str = None):
    """
    Process all FFN files in a directory

    Args:
        input_dir: Directory containing FFN files
        output_dir: Directory to save TSV files
        cache_dir: Directory for cache files (optional)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = output_path / "cache"

    # Initialize components
    logger.info("Initializing GenomeOcean embedder...")
    embedder = GenomeOceanEmbedder()

    logger.info("Initializing embedding cache...")
    cache = EmbeddingCache(cache_dir)

    # Find all FFN files
    ffn_files = list(input_path.glob("*.ffn"))
    logger.info(f"Found {len(ffn_files)} FFN files to process")

    if not ffn_files:
        logger.warning(f"No FFN files found in {input_dir}")
        return

    # Process each file
    for ffn_file in ffn_files:
        logger.info(f"Processing {ffn_file.name}...")

        # Read sequences
        sequences = read_ffn_file(str(ffn_file))
        if not sequences:
            logger.warning(f"No sequences found in {ffn_file.name}")
            continue

        # Generate embeddings
        embeddings = process_sequences_in_batches(sequences, embedder, cache)

        # Save to TSV
        output_file = output_path / f"{ffn_file.stem}_embeddings.tsv"
        save_embeddings_to_tsv(embeddings, str(output_file))

        # Save cache periodically
        cache.save()

    # Final cache save
    cache.save()
    logger.info("Processing complete!")


def main():
    """Main function"""
    # Default paths
    default_input = "/Users/danielkorkin/Documents/SMTB2025/Lab16/SMTB2025_AMR/AMR-DL/Data/Raw/pa_ffn"
    default_output = "/Users/danielkorkin/Documents/SMTB2025/Lab16/SMTB2025_AMR/AMR-DL/Data/Embeddings"

    # Parse command line arguments
    if len(sys.argv) >= 2:
        input_dir = sys.argv[1]
    else:
        input_dir = default_input

    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        output_dir = default_output

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    try:
        process_ffn_directory(input_dir, output_dir)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
