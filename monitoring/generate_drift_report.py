import pandas as pd
from pathlib import Path

from evidently.report import Report
from evidently.metrics import DataDriftTable, ColumnDriftMetric

def main():
    base_path = Path("monitoring")
    report_dir = base_path / "reports"
    report_dir.mkdir(exist_ok=True)

    reference = pd.read_csv(base_path / "reference_data.csv")
    current = pd.read_csv(base_path / "current_data.csv")

    report = Report(
        metrics=[
            DataDriftTable(),
            ColumnDriftMetric(column_name="target")
        ]
    )

    report.run(
        reference_data=reference,
        current_data=current
    )

    report_path = report_dir / "drift_report.html"
    report.save_html(str(report_path))

    print(f"Drift report generated at: {report_path}")

if __name__ == "__main__":
    main()

 