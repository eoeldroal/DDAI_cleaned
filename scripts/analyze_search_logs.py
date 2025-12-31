import re
import os

LOG_FILE = "wandb/latest-run/files/output.log"

def analyze_logs():
    if not os.path.exists(LOG_FILE):
        print(f"Error: {LOG_FILE} not found.")
        return

    print(f"Analyzing {LOG_FILE} using <think> tag markers...")
    
    samples = []
    current_sample = []
    
    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # Start of a new thought block often implies a new turn or sample
            if "<think>" in line:
                if current_sample:
                    samples.append("\n".join(current_sample))
                    current_sample = []
            
            if current_sample or "<think>" in line:
                current_sample.append(line)
        
        # Add last sample
        if current_sample:
            samples.append("\n".join(current_sample))

    print(f"Found {len(samples)} potential sample traces.")

    truncated_count = 0
    long_query_count = 0
    format_error_count = 0
    garbage_count = 0
    
    long_query_threshold = 300 

    print("\n--- Analysis Results ---\n")

    for i, resp in enumerate(samples):
        # 1. Check for Truncation
        # Since we are splitting by <think>, a single sample might be just one turn.
        # But if we see <search_complete> it means the task finished successfully.
        # If a sequence of thoughts ends without <search_complete>, it MIGHT be truncated.
        # However, since we split by <think>, we might be breaking a single multi-turn sample into pieces.
        # Let's verify if 'search_complete' exists in THIS block.
        has_complete = "<search_complete>true</search_complete>" in resp
        
        # 2. Check for Long Queries
        search_tags = re.findall(r"<search>(.*?)</search>", resp, re.DOTALL)
        for query in search_tags:
            if len(query) > long_query_threshold:
                long_query_count += 1
                # print(f"[Long Query] Sample {i} (Len: {len(query)}): {query[:50]}...")
            
            if "I need to" in query or "because" in query:
                 format_error_count += 1

        # 3. Check for Garbage (Model Collapse)
        # Count non-ascii characters
        non_ascii = sum(1 for c in resp if ord(c) > 127)
        if non_ascii > 50: # Threshold for garbage
            garbage_count += 1
            print(f"[Garbage Detect] Sample {i} has {non_ascii} non-ascii chars. Start: {resp[:50]}...")

    print(f"\nTotal 'Think' Blocks Analyzed: {len(samples)}")
    print(f"Excessively Long Queries (> {long_query_threshold} chars): {long_query_count}")
    print(f"Dirty Queries (Reasoning mixed in): {format_error_count}")
    print(f"Garbage / Model Collapse Cases: {garbage_count}")

if __name__ == "__main__":
    analyze_logs()
