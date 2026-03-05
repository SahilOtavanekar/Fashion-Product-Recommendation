import pickle
import os

# Load the filenames
with open('filenames.pkl', 'rb') as f:
    filenames = pickle.load(f)

# Display first few filenames to see their format
print("First few filenames:")
for i in range(min(5, len(filenames))):
    print(f"  {filenames[i]}")

# Fix the paths - convert from /content/images/ to local images/
fixed_filenames = []
for fname in filenames:
    # Extract just the filename (e.g., "12345.jpg")
    base_filename = os.path.basename(fname)
    # Create local path
    local_path = os.path.join("images", base_filename)
    fixed_filenames.append(local_path)

# Save the fixed filenames
with open('filenames.pkl', 'wb') as f:
    pickle.dump(fixed_filenames, f)

print(f"\nFixed {len(fixed_filenames)} filenames")
print("First few fixed filenames:")
for i in range(min(5, len(fixed_filenames))):
    print(f"  {fixed_filenames[i]}")

print("\nVerifying files exist:")
for i in range(min(5, len(fixed_filenames))):
    exists = os.path.exists(fixed_filenames[i])
    print(f"  {fixed_filenames[i]}: {exists}")
