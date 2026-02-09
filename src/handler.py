import time
import gc
import torch
import runpod

from gliner2 import GLiNER2

llm = None

def initialize_model():
	global llm
	if llm is None:
		try:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			llm = GLiNER2.from_pretrained("fastino/gliner2-large-v1").to(device)
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