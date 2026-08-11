from pathlib import Path
from shutil import copy2

from ultralytics import YOLO


def main():
    training_directory = Path(__file__).resolve().parent
    project_directory = training_directory.parent

    dataset_yaml = training_directory / "data.yaml"
    pretrained_model = project_directory / "yolo11n.pt"
    output_directory = project_directory / "runs"
    api_model_directory = project_directory / "models"

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Missing dataset file: {dataset_yaml}")

    if not pretrained_model.exists():
        raise FileNotFoundError(f"Missing pretrained model: {pretrained_model}")

    model = YOLO(str(pretrained_model))

    model.train(
        data=str(dataset_yaml),
        epochs=100,
        imgsz=640,
        batch=8,
        device="cpu",  # Change to 0 for an NVIDIA GPU
        workers=2,
        patience=30,
        pretrained=True,
        plots=True,
        save=True,
        project=str(output_directory),
        name="taka_yolo11",
        exist_ok=True,
    )

    trained_model = (
        output_directory
        / "taka_yolo11"
        / "weights"
        / "best.pt"
    )

    if trained_model.exists():
        api_model_directory.mkdir(parents=True, exist_ok=True)
        destination = api_model_directory / "best.pt"
        copy2(trained_model, destination)

        print("\nTraining completed successfully.")
        print(f"Generated model: {trained_model}")
        print(f"Copied API model: {destination}")
    else:
        print(f"Training finished, but best.pt was not found: {trained_model}")


if __name__ == "__main__":
    main()