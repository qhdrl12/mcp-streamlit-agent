.PHONY: run setup clean

# Run the Streamlit app
run:
	streamlit run app.py

# Install dependencies
setup:
	pip install -r requirements.txt
	cp -n .env.example .env || true

# Clean up generated files
clean:
	rm -rf __pycache__
	rm -rf models/__pycache__
	rm -rf adapters/__pycache__
	rm -rf utils/__pycache__

# Create default configs directory if it doesn't exist
configs:
	mkdir -p configs

# Initialize a new configuration
init: configs
	cp -n configs/default.json configs/custom.json || true 