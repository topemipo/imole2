import os
import json
from dotenv import load_dotenv
import anthropic
from functions import summarization_pipeline

# Load environment variables
load_dotenv()

# Anthropic
anthropic_key = os.getenv("ANTHROPIC_KEY")
client_atp = anthropic.Anthropic(api_key=anthropic_key)

# Folder containing documents
document_folder = "chosen_50"

# Output JSON file
summary_file = "summaries.json"
error_log_file = "error_log.txt"

# Dictionary to store summaries
summaries = {}
errors = []

# Iterate over each document in the folder
for i, filename in enumerate(os.listdir(document_folder)):
    file_path = os.path.join(document_folder, filename)
    try:
        # Read document text with error handling
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Generate summary
        print(f"Summarizing {filename} ({i+1}/50)...")
        summary = summarization_pipeline(text)

        # Store summary with metadata
        summaries[f"document_{i+1}"] = {
            "filename": filename,
            "original_length": len(text),
            "summary": summary
        }
    
    except Exception as e:
        # Log errors and continue
        error_message = f"Error processing {filename}: {str(e)}"
        print(error_message)
        errors.append(error_message)

# Save summaries to JSON
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=4)

# Save errors to a log file
if errors:
    with open(error_log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(errors))
    print(f"Some files failed to process. See {error_log_file} for details.")

print(f"Summaries saved to {summary_file}")
