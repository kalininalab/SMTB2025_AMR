# Genome Ocean Embeddings Extraction

1. Ensure you have raw ffns
    - Run annotation with PROKKA with species specific data created from uniprot proteins of this species
2. Use `smtb_amr/features/genome_ocean/rm_useless_parts.py` to find useful parts relating to AMR and isolate them
3. Use `smtb_amr/features/genome_ocean/embeddings.py` to extract embeddings from the filtered ffn files into tsv form
4. Use `smtb_amr/features/genome_ocean/average_embeddings.py` to average out the embeddings
5. Use `smtb_amr/features/genome_ocean/sorting.py` to put it into respective bacteria folders