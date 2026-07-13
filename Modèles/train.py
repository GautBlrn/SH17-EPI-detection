from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    model = YOLO(str(ROOT / "models" / "yolo11l.pt"))
    model.train(
        data=str(ROOT / "SH17_yolo" / "data.yaml"),
        imgsz=768,
        epochs=100,
        batch=4,
        cache=False,
        workers=4,
        optimizer="AdamW",
        lr0=8e-05,
        cos_lr=True,
        patience=20,
        seed=42,
        project=str(ROOT / "runs" / "train"), name="yolo11l_full",
    )


if __name__ == "__main__":
    main()
