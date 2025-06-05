import numpy as np

def calculate_rmse(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize lists for labels and predictions
    labels = []
    predictions = []

    # Skip the header
    for i, line in enumerate(lines):
        if i == 0:
            continue  # Skip header line
        try:
            parts = line.strip().split('\t')
            # Ensure there are at least 4 columns
            if len(parts) >= 4:
                label = float(parts[2])
                prediction = float(parts[3])
                labels.append(label)
                predictions.append(prediction)
            else:
                print(f"Skipping malformed line: {line.strip()}")
        except ValueError:
            print(f"Skipping invalid line: {line.strip()}")

    # Convert to numpy arrays
    labels = np.array(labels)
    predictions = np.array(predictions)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((labels - predictions) ** 2))
    return rmse

# Path to your TSV file
file_path = 'Test_results.txt'

# Calculate and print RMSE
rmse = calculate_rmse(file_path)
print(f"Root Mean Square Error (RMSE): {rmse}")
