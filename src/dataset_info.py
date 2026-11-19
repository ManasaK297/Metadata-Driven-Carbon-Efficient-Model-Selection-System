"""Print dataset folder, filename, and dimensions (rows, columns) for each CSV."""

from pathlib import Path
import pandas as pd


def main():
    data_root = Path(__file__).parent.parent / "Data"
    
    if not data_root.exists():
        print(f"Data folder not found: {data_root}")
        return
    
    # Find all CSV files
    csv_files = sorted(data_root.rglob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in Data/")
        return
    
    print(f"{'Folder':<50} {'Filename':<40} {'Rows':<10} {'Columns':<10}")
    print("=" * 110)
    
    for csv_path in csv_files:
        # Get folder name (relative to Data/)
        try:
            folder = str(csv_path.parent.relative_to(data_root))
        except ValueError:
            folder = str(csv_path.parent)
        
        # Get filename
        filename = csv_path.name
        
        # Get rows and columns
        try:
            df = pd.read_csv(csv_path)
            rows = len(df)
            cols = len(df.columns)
        except Exception as e:
            rows = f"Error: {e}"
            cols = "N/A"
        
        print(f"{folder:<50} {filename:<40} {str(rows):<10} {str(cols):<10}")


if __name__ == "__main__":
    main()
