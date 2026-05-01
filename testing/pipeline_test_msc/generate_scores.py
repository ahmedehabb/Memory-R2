import json

import pandas as pd

# Category id to name mapping
def get_category_name(cat_id):
    mapping = {
        1: "single-hop",
        2: "temporal",
        3: "multi-hop",
        4: "open-domain"
    }
    return mapping.get(cat_id, str(cat_id))

# Load the evaluation metrics data
with open("evaluation_metrics.json", "r") as f:
    data = json.load(f)

# Flatten the data into a list of question items
all_items = []
for key in data:
    all_items.extend(data[key])

# Convert to DataFrame
df = pd.DataFrame(all_items)

# Convert category to numeric type
# df["category"] = pd.to_numeric(df["category"])

# Map category id to name
df["category_name"] = df["category"]

# Calculate mean scores by category name
result = df.groupby("category_name").agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

# Add count of questions per category name
result["count"] = df.groupby("category_name").size()

# Reorder and filter the result
order = ["single-session-user", "single-session-preference", "single-session-assistant", "multi-session", "knowledge-update", "temporal-reasoning"]
result = result.loc[[cat for cat in order if cat in result.index]]

# Reset index so category_name is a column
result = result.reset_index()

# Print the results
print("Mean Scores Per Category:")
print(result)

# Calculate overall means
overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

print("\nOverall Mean Scores:")
print(overall_means)
