import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    print("Starting data ingestion process...\n")
    
    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        print("\n✓ Data ingestion completed successfully!")
        print(f"✓ Training data saved to: {train_path}")
        print(f"✓ Test data saved to: {test_path}")
        
    except Exception as e:
        print(f"✗ Error during data ingestion: {e}")
        sys.exit(1)
