# Отчет по ЭВМ: задача Неймана

report.pdf - готовый отчет.
main.py - программа для расчетов и построения графиков.
report.tex - LaTeX-исходник отчета.
test_main.py - короткие проверки кода.
requirements.txt - Python-зависимости.
output/figs - графики, использованные в отчете.
output/data - таблицы с результатами расчетов.

# чтобы запустить

python -m pip install -r requirements.txt
python main.py --out output
python -m unittest test_main.py
