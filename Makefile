.PHONY: up setup setup-lite demo status eval test down
up:          ## start pgvector + MySQL(Sakila)
	docker compose up -d
setup:       ## init pgvector, load schema cards, ingest ./pdfs (with diagram model)
	python -m kb setup
setup-lite:  ## same but without any vision model (works on a weak laptop)
	python -m kb setup --no-vision
demo:        ## open the web UI
	streamlit run app.py
status:
	python -m kb status
eval:
	python -m kb eval
test:
	python -m unittest discover -s tests -v
down:
	docker compose down
