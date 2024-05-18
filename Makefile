.PHONY: test

test: pre-commit pytest

pytest:
	poetry run pytest

pre-commit:
	poetry run pre-commit run --all
