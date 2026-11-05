from pathlib import Path

from ultralytics import YOLO

# Корінь проєкту = на рівень вище від папки src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

model = YOLO("yolo11n.pt")

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.25,
    save=True,
    project=str(PROJECT_ROOT / "runs"),  # абсолютний шлях, щоб Ultralytics не додавав runs/detect/
    name="smoke_predict",
)

print("Smoke test completed.")
print(f"Number of result objects: {len(results)}")
