import os

def concatenate_files(source_dir, output_file, skip_paths=None):
    """
    Concatenate the contents of Python files in the source directory into a single text file.
    Supports skipping specific files or entire directories.

    Args:
        source_dir (str): The source directory to scan for Python files.
        output_file (str): The path to the output file where concatenated content will be saved.
        skip_paths (list): A list of file or directory paths to skip (relative to the source_dir).
    """
    if skip_paths is None:
        skip_paths = []

    # Normalize skip paths to absolute paths for easier comparison
    skip_paths_abs = [os.path.abspath(os.path.join(source_dir, path)) for path in skip_paths]

    with open(output_file, 'w') as outfile:
        for root, dirs, files in os.walk(source_dir):
            # Convert current root to absolute path for comparison
            root_abs = os.path.abspath(root)

            # Determine if current directory or any parent directory is in skip_paths
            skip_current_dir = False
            for skip_path in skip_paths_abs:
                if os.path.commonpath([root_abs, skip_path]) == skip_path:
                    skip_current_dir = True
                    break
            if skip_current_dir:
                # Skip processing this directory and its subdirectories
                dirs[:] = []  # Clear dirs to prevent walking into subdirectories
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_path_abs = os.path.abspath(file_path)

                    # Check if the file is in the skip list
                    skip_file = False
                    for skip_path in skip_paths_abs:
                        if os.path.commonpath([file_path_abs, skip_path]) == skip_path:
                            skip_file = True
                            break
                    if skip_file:
                        continue

                    # Compute the relative path for markers
                    relative_path = os.path.relpath(file_path, source_dir)

                    with open(file_path, 'r', encoding='utf-8') as infile:
                        file_content = infile.read()

                    # Write the clear separator and file path
                    separator = "\n" + "=" * 30 + "\n"
                    outfile.write(separator)
                    outfile.write(f"# Begin file: {relative_path}\n")
                    outfile.write(file_content)
                    outfile.write(f"\n# End file: {relative_path}\n")
                    outfile.write(separator)

    print(f"Concatenation completed. Output saved to '{output_file}'.")

if __name__ == "__main__":
    # Configuration
    source_directory = "src"  # Path to the source directory
    output_filename = "concatenated_code_prompt.txt"  # Output file path
    skip_paths = [
        # Add any files or directories to skip here, relative to the source_directory
        # Example: "pmbrl/envs/envs/assets"
        # Example: "pmbrl/__pycache__"
        "pmbrl/control/__init__.py",
        "pmbrl/models/__init__.py",
        "pmbrl/training/__init__.py",
        "pmbrl/logger/__init__.py",
        "pmbrl/utils/",
        "pmbrl/configs.py",
        "scripts/__init__.py",
        "pmbrl/envs",
    ]

    # Execute the concatenation
    concatenate_files(source_directory, output_filename, skip_paths=skip_paths)
