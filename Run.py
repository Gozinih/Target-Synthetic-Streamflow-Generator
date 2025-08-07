import os
import subprocess
import sys

# === Step 1: Define paths ===
generator_dir = os.path.join(os.getcwd(), "GeneratorCodes")
main_file = os.path.join(generator_dir, "a1_Main.py")
user_input_file = os.path.join(os.getcwd(), "InputData.txt")

# === Step 2: Confirm paths exist ===
if not os.path.exists(main_file):
    print("❌ ERROR: a1_Main.py not found in 'GeneratorCodes' folder.")
    input("Press Enter to exit...")
    sys.exit(1)

if not os.path.exists(user_input_file):
    print("❌ ERROR: InputData.txt not found in the current directory.")
    input("Press Enter to exit...")
    sys.exit(1)

# === Step 3: Install missing packages ===
required_packages = ['pandas', 'numpy', 'h5py', 'openpyxl', 'scipy', 'joblib', 'matplotlib', 'numba', 'tqdm']

for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"🔧 Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# === Step 4: Replace section between '# start' and '# end' ===
with open(user_input_file, "r", encoding="utf-8") as f:
    new_input_block = f.read().splitlines()

with open(main_file, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

start_index = None
end_index = None

for i, line in enumerate(lines):
    if line.strip().lower() == "# start":
        start_index = i
    elif line.strip().lower() == "# end":
        end_index = i
        break

if start_index is None or end_index is None or end_index <= start_index:
    print("❌ ERROR: Could not find '# start' and '# end' markers in a1_Main.py.")
    input("Press Enter to exit...")
    sys.exit(1)

new_content = lines[:start_index + 1] + new_input_block + lines[end_index:]

with open(main_file, "w", encoding="utf-8") as f:
    f.write('\n'.join(new_content))

print("✅ User Input Data Is Loaded")

# === Step 5: Run the main script ===

subprocess.run([sys.executable, main_file])
