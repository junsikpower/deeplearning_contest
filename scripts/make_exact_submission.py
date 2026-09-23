from pathlib import Path

from orbit_predict.data import DatasetPaths, load_datasets, make_submission
from orbit_predict.model_a import CircleParameters, predict_circle


def main() -> None:
    data = load_datasets(
        DatasetPaths(
            train=Path("orbit-predict-satellite-position/Train_DS..csv"),
            test=Path("orbit-predict-satellite-position/Test_DS..csv"),
            sample_submission=Path(
                "orbit-predict-satellite-position/Sample_Submission_DS..csv"
            ),
        )
    )
    parameters = CircleParameters(
        center_x=630.0,
        center_y=550.0,
        radius=420.0,
        coefficient_d=-1260.0,
        coefficient_e=-1100.0,
        coefficient_f=523000.0,
    )
    result = predict_circle(data.test, parameters)
    if result.exceptions:
        raise RuntimeError(f"모델 A 예외 행 {len(result.exceptions)}개를 확인해야 합니다.")
    submission = make_submission(data.sample_submission, result.predictions)
    path = Path("outputs/submission_model_a_exact.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    print(f"{path}: {len(submission):,}행")


if __name__ == "__main__":
    main()
