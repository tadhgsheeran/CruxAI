# CruxAI

CruxAI is an AI-powered MoonBoard climbing analysis project.

The current application uses a PyTorch neural network and a FastAPI backend to predict the difficulty grade of a MoonBoard route represented as an 18 × 11 hold matrix.

## Current Features

- PyTorch MoonBoard grade-prediction model
- FastAPI prediction endpoint
- Input validation using Pydantic
- Health-check endpoint
- Automated API tests using pytest
- Saved model checkpoint loading

## API Endpoints

### `GET /`

Confirms that the API is running.

### `GET /health`

Checks the health of the application and confirms that the model loaded.

### `POST /predict-grade`

Accepts an 18 × 11 MoonBoard route matrix and returns:

- raw predicted grade
- rounded grade
- formatted V-grade
- model version

## Run Locally

Create and activate the environment, then install the dependencies:

```bash
pip install -r requirements.txt