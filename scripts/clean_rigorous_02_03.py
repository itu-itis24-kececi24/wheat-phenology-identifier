import os
import shutil

# Cross-platform paths (works on both Windows and Linux)
# We use forward slashes and replace them with the OS-specific separator at runtime
DATA_DIR = os.path.join(".", "data")
SOURCE_DIR = os.path.join(DATA_DIR, "02.03")
DEST_DIR = os.path.join(DATA_DIR, "bad_images", "02.03")

# List of 16 verified bad images from Camera K1
BAD_IMAGES = [
    "2013/K1/10X/02_03-2013_04_11-10_00-K1-10X.jpeg",
    "2014/K1/10X/02_03-2014_03_10-09_56-K1-10X.jpeg",
    "2014/K1/1X/02_03-2014_03_04-10_49-K1-1X.jpeg",
    "2014/K1/1X/02_03-2014_03_06-10_14-K1-1X.jpeg",
    "2014/K1/1X/02_03-2014_03_10-09_58-K1-1X.jpeg",
    "2014/K1/1X/02_03-2014_11_22-10_01-K1-1X.jpeg",
    "2015/K1/10X/02_03-2015_02_18-10_30-K1-10X.jpeg",
    "2015/K1/10X/02_03-2015_02_21-10_30-K1-10X.jpeg",
    "2015/K1/10X/02_03-2015_02_23-10_07-K1-10X.jpeg",
    "2015/K1/10X/02_03-2015_03_12-10_00-K1-10X.jpeg",
    "2015/K1/10X/02_03-2015_03_24-10_00-K1-10X.jpeg",
    "2015/K1/10X/02_03-2015_05_19-08_00-K1-10X.jpeg",
    "2015/K1/10X/02_03-2015_05_31-10_25-K1-10X.jpeg",
    "2015/K1/1X/02_03-2015_02_18-10_30-K1-1X.jpeg",
    "2015/K1/1X/02_03-2015_03_02-10_00-K1-1X.jpeg",
    "2015/K1/1X/02_03-2015_03_12-10_00-K1-1X.jpeg"
]

def clean():
    moved_count = 0
    errors = 0
    
    # Convert directories to absolute paths for safe execution
    abs_source = os.path.abspath(SOURCE_DIR)
    abs_dest = os.path.abspath(DEST_DIR)
    
    print(f"Moving 16 verified bad images from:")
    print(f"  Source: {abs_source}")
    print(f"  Dest:   {abs_dest}\n")
    
    for rel_path in BAD_IMAGES:
        # Convert path separators from forward slash to OS-specific separator
        os_rel_path = rel_path.replace("/", os.sep)
        src_path = os.path.join(abs_source, os_rel_path)
        dest_path = os.path.join(abs_dest, os_rel_path)
        
        if not os.path.exists(src_path):
            # Already moved or does not exist
            continue
            
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.move(src_path, dest_path)
            moved_count += 1
        except Exception as e:
            print(f"Error moving {os_rel_path}: {e}")
            errors += 1
            
    print("\n" + "="*50)
    print("           RIGOROUS CLEANUP COMPLETE")
    print("="*50)
    print(f"Successfully moved: {moved_count} images.")
    if errors > 0:
        print(f"Encountered errors: {errors} files.")
    print("="*50)

if __name__ == "__main__":
    clean()
