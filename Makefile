lint: isort-check black-check flake8-check mypy-check spell-check

isort-check:
	poetry run isort --sl -c src/

black-check:
	poetry run black --check src/

flake8-check:
	poetry run flake8 src/ --ignore=D101,D102

mypy-check:
	poetry run mypy

spell-check:
	poetry run codespell src/ *.md

format:
	poetry run isort --sl src/
	poetry run black src/
	poetry run flake8 src/ --ignore=D101,D102
	poetry run mypy
