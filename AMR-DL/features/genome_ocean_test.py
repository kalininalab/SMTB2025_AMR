# Load model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "pGenomeOcean/GenomeOcean-4B",
    trust_remote_code=True,
    padding_side="left",
)
model = AutoModelForCausalLM.from_pretrained(
    "pGenomeOcean/GenomeOcean-4B",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to("cuda")
# Embedding
sequences = [
    "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT",
    "CAGTCAGTGGCTAGCATGCTAGCATCGATCGATCGATCGATCGATCGATCGATCGGTGCATGCTAGCATCGATCGATCGAA",
]
output = tokenizer.batch_encode_plus(
    sequences, max_length=10240, return_tensors="pt", padding="longest", truncation=True
)
input_ids = output["input_ids"].cuda()
attention_mask = output["attention_mask"].cuda()
model_output = (
    model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()
)
attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
embedding = torch.sum(model_output * attention_mask, dim=1) / torch.sum(
    attention_mask, dim=1
)
print(f"Shape: {embedding.shape}")  # (2, 3072)
print(f"Embedding: {embedding}")
