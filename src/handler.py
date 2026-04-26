import os
import time
import gc
import torch
import runpod

from gliner2 import GLiNER2

HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
llm = None

def resolve_snapshot_path(model_id: str) -> str:
	"""
	Resolve the local snapshot path for a cached model.

	Args:
		model_id: The model name from Hugging Face
			(e.g., 'distilbert/distilbert-base-uncased-finetuned-sst-2-english')

	Returns:
		The full path to the cached model snapshot
	"""
	if "/" not in model_id:
		raise ValueError(f"model_id '{model_id}' must be in 'org/name' format")
	
	org, name = model_id.split("/", 1)
	model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
	refs_main = os.path.join(model_root, "refs", "main")
	snapshots_dir = os.path.join(model_root, "snapshots")
	
	# Read the snapshot hash from refs/main
	if os.path.isfile(refs_main):
		with open(refs_main, "r") as f:
			snapshot_hash = f.read().strip()
		candidate = os.path.join(snapshots_dir, snapshot_hash)
		if os.path.isdir(candidate):
			return candidate
	
	# Fall back to first available snapshot
	if os.path.isdir(snapshots_dir):
		versions = [
			d for d in os.listdir(snapshots_dir)
			if os.path.isdir(os.path.join(snapshots_dir, d))
		]
		if versions:
			versions.sort()
			return os.path.join(snapshots_dir, versions[0])
	
	raise RuntimeError(f"Cached model not found: {model_id}")

def initialize_model():
	LOCAL_PATH = resolve_snapshot_path(
		"fastino/gliner2-large-v1"
	)
	
	global llm
	if llm is None:
		try:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			llm = GLiNER2.from_pretrained(LOCAL_PATH).to(device)
		except Exception as e:
			print(f"Error loading model: {str(e)}")
			raise
	return llm

#   This function processes incoming requests to your Serverless endpoint.
#
#    Args:
#        event (dict): Contains the input data and request metadata
#
#    Returns:
#       Any: The result to be returned to the client
def handler(event):
	input = event['input']
	
	texts = input.get('texts')
	
	if not (isinstance(texts, str) or isinstance(texts, list)):
		return {
			"error": "'input' must be a string or list of strings"
		}
	
	# Convert input to list format
	if isinstance(texts, str):
		texts = [texts]
	
	if len(texts) == 0:
		return {
			"error": "Empty input"
		}
	
	# Validate all inputs are strings
	if not all(isinstance(text, str) for text in texts):
		return {
			"error": "All inputs must be strings"
		}
	
	model = initialize_model()
	
	entity_schema = {
		"medication": "Names of drugs, medications, or pharmaceutical substances",
		"symptom": "Medical symptoms, conditions, or patient complaints",
		"person": "Names of people",
		"organization": "A company, institution, or group formed for a specific purpose",
		"product": "An item or service offering value and created for sale or use",
		"location": "Company headquarters or office location",
		"event": "A specific or noteworthy instance, or activity occurring within a defined context"
	}
	
	results = []
	
	start_time = time.time()
	for text in texts:
		entities = model.extract_entities(text, entity_schema, threshold=0.9)
		results.append(entities)
	inference_time = time.time() - start_time
	
	print(f"Generated {len(texts)} entities in {inference_time:.2f}s")
	
	gc.collect()
	
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		
		# Reset CUDA device to fully clear memory
		torch.cuda.reset_peak_memory_stats()
		torch.cuda.synchronize()  # Wait for all streams on the current device
	
	return results

# Start the Serverless function when the script is run
if __name__ == '__main__':
	runpod.serverless.start({
		'handler': handler
	})