"""Запуск VoxMatch. Файл намеренно пустой на верхнем уровне:
на Windows рабочие процессы анализа импортируют его и не должны грузить окно."""

if __name__ == "__main__":
    import multiprocessing
    import os
    import sys

    multiprocessing.freeze_support()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import voxmatch_app

    voxmatch_app.main()
