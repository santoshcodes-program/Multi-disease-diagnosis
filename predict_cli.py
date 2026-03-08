import argparse
import json

from src.predictor import MultiDiseasePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-disease prediction from CLI")
    parser.add_argument("--symptoms", type=str, required=True, help="Comma-separated symptoms")
    parser.add_argument("--age", type=float, default=35.0)
    parser.add_argument("--gender", type=str, default="male")
    parser.add_argument("--blood-pressure", type=float, default=120.0)
    parser.add_argument("--glucose", type=float, default=100.0)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    predictor = MultiDiseasePredictor()
    symptoms = [s.strip() for s in args.symptoms.split(",") if s.strip()]
    clinical_features = {
        "age": args.age,
        "gender": 1 if args.gender.strip().lower() == "male" else 0,
        "blood_pressure": args.blood_pressure,
        "glucose": args.glucose,
    }
    predictions = predictor.predict_top_k(symptoms=symptoms, clinical_features=clinical_features, top_k=args.top_k)
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()

