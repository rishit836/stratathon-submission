# Off-Road Segmentation Web Application

This is a Flask-based web application that uses a deep learning model to perform semantic segmentation on off-road images. It identifies various terrain types such as trees, bushes, grass, rocks, and sky.
The model is built using a DINOv2 backbone with a custom Enhanced Segmentation Head.

## Features

-   **User-friendly Interface**: Simple drag-and-drop or file selection for image uploads.
-   **AI Inference**: Uses a fine-tuned PyTorch model for accurate segmentation.
-   **Visual Results**: Displays the original image alongside the predicted segmentation mask.
-   **Downloadable Results**: (Implied functionality) Processed images are saved in `static/results`.

## Directory Structure

```
frontend/
├── main.py                 # The main Flask application entry point.
├── model/                  # Directory containing the trained model weights.
│   ├── best_finetuned_backbone.pth
│   └── best_segmentation_head.pth
├── static/                 # Static assets.
│   ├── css/                # Stylesheets.
│   ├── results/            # Stores processed segmentation results.
│   └── uploads/            # Temporary storage for uploaded images.
├── templates/              # HTML templates.
│   └── index.html          # The main page template.
└── README.md               # This documentation file.
```

## Prerequisites

-   Python 3.8 or higher
-   pip (Python package manager)
-   Optional: CUDA-capable GPU for faster inference (the app automatically selects CPU or GPU).

## Installation

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    You will need the following libraries:
    - Flask
    - torch
    - torchvision
    - numpy
    - opencv-python
    
    Run the following command:
    ```bash
    pip install flask torch torchvision numpy opencv-python
    ```

## Running the Application Locally

1.  **Start the Flask server:**
    ```bash
    python main.py
    ```

2.  **Access the application:**
    Open your web browser and go to:
    [http://localhost:5000](http://localhost:5000)

## Deployment

To deploy this application to a production environment (e.g., Render, Heroku, AWS, DigitalOcean), follow these general steps:

### 1. Prepare `requirements.txt`
Generate a list of dependencies so the hosting provider knows what to install.
```bash
pip freeze > requirements.txt
```
*Note: Ensure `torch` uses the CPU version for cloud platforms with limited slug size, unless you are using a GPU instance.*

### 2. Create a `Procfile` (for Heroku/Render)
Create a file named `Procfile` (no extension) in the `frontend` folder with the following content:
```
web: gunicorn main:app
```
You will need to install `gunicorn`:
```bash
pip install gunicorn
```

### 3. Environment Variables
If your cloud provider requires it, set the `FLASK_APP` environment variable to `main.py`.

### 4. Special Considerations for PyTorch
PyTorch is a large library. If you encounter size limits during deployment (e.g., on Heroku free tier), strictly install the CPU-only version in your `requirements.txt` by using the `--extra-index-url` or explicitly specifying the wheel, or use a Docker container.

**Example Dockerfile (Optional):**
If deploying via Docker, create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]
```

## detailed Model Information

-   **Backbone**: DINOv2 (ViT-S/14) - Fine-tuned.
-   **Head**: Custom Enhanced Segmentation Head (ASPP + Decoder).
-   **Classes**:
    0. Background
    1. Trees
    2. Lush Bushes
    3. Dry Grass
    4. Dry Bushes
    5. Ground Clutter
    6. Logs
    7. Rocks
    8. Landscape
    9. Sky

## Troubleshooting

-   **Model File Not Found**: Ensure `best_finetuned_backbone.pth` and `best_segmentation_head.pth` are present in the `frontend/model/` directory.
-   **CUDA/GPU Errors**: If you don't have a GPU, the app should automatically fall back to CPU. Ensure you have the correct version of PyTorch installed for your system.
