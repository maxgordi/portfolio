"""Запуск окна «VocalSync». Файл намеренно пустой на верхнем уровне:
рабочие процессы анализа высоты импортируют его и не должны грузить окно."""

if __name__ == "__main__":
    import multiprocessing
    import os
    import sys

    multiprocessing.freeze_support()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vocalsync import ui

    ui.main()
