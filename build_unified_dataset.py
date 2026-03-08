from pathlib import Path

from src.data_integration import load_and_integrate_datasets


def main() -> None:
    out_dir = Path("data") / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    integrated = load_and_integrate_datasets()
    unified = integrated.symptoms_df
    output_path = out_dir / "unified_symptom_dataset.csv"
    unified.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {unified.shape[0]}")
    print(f"Columns: {unified.shape[1]}")
    print(f"Symptoms: {unified.shape[1] - 1}")
    print(f"Diseases: {unified['disease'].nunique()}")


if __name__ == "__main__":
    main()

