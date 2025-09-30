# CartPoleDL

## Setup Instructions

### 1. Create a Virtual Environment
First, create a Python virtual environment:
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

#### Install PyTorch
**Windows (with CUDA support):**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**macOS (default package):**
```bash
pip3 install torch torchvision
```

#### Install Other Requirements
```bash
pip install -r requierments.txt
```

### 4. Run Jupyter Notebook
After installing all dependencies, you can start the Jupyter notebook:
```bash
jupyter notebook
```

Then open the `cartPole.ipynb` file to run the CartPole deep learning experiments.