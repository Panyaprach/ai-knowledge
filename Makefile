.PHONY: up setup setup-lite demo status eval test down
up:
	docker compose up -d
setup:
	python -m kb setup
setup-lite:
	python -m kb setup --no-vision
demo:
	streamlit run app.py
status:
	python -m kb status
eval:
	python -m kb eval
test:
	python -m unittest discover -s tests -v
down:
	docker compose down
