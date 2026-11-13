"""Tham số cá nhân hóa theo mã sinh viên (đề thi giữa kỳ).

N          = 200 + 20a
n_features >= 8 + b
n_outliers >= 5 + 2c
a = s mod 5; b = (s div 10) mod 5; c = (s div 100) mod 5
"""

STUDENT_ID = 24022433
SEED = 42


def derive(s: int) -> dict:
    a = s % 5
    b = (s // 10) % 5
    c = (s // 100) % 5
    return {
        "a": a,
        "b": b,
        "c": c,
        "N": 200 + 20 * a,
        "min_features": 8 + b,
        "min_outliers": 5 + 2 * c,
    }


if __name__ == "__main__":
    if STUDENT_ID is None:
        raise SystemExit("Chưa điền STUDENT_ID trong src/params.py")
    print(derive(STUDENT_ID))
