# DeepPrune

## Project Description
DeepPrune is a tool designed to optimize the **photogrammetry workflow** by intelligently selecting the most relevant images from large datasets. In photogrammetry, a high number of overlapping images is essential for accurate 3D reconstruction, but excessive redundancy increases processing time and computational costs.

DeepPrune leverages **Convolutional Neural Networks (CNNs)** to analyze and filter images based on **overlap, perspective, and coverage**, ensuring that only the most useful images are used for 3D modeling. This significantly speeds up processing in software like **[Reality Capture](https://www.capturingreality.com/)** while maintaining reconstruction quality.

**Key Features:**
- DeepPrune is designed to train a CNN to determine the minimum number of images needed to ensure proper alignment without compromising quality.

- Instead of relying on fixed overlap percentages, it prioritizes images with the highest geometric information and the least redundancy.

- By implementing intelligent image selection based on overlap, DeepPrune could reduce the total number of images by 30%-50% while maintaining the same level of accuracy in 3D reconstruction.

Essentially, **DeepPrune** is a CNN-based model designed to identify and discard unnecessary images in 3D reconstruction, ensuring sufficient overlap while maintaining quality.

---

## Set up your Environment

### **`macOS`** type the following commands : 

- For installing the virtual environment you can either use the [Makefile](Makefile) and run `make setup` or install it manually with the following commands:

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```
Or ....
- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```

---

## Running with Docker

If you prefer to use Docker instead of setting up a local Python environment, you can run DeepPrune in a containerized environment.

### Prerequisites
- Docker installed on your computer ([Download Docker](https://www.docker.com/get-started))

### Quick Start

1. **Build the Docker image:**
   ```bash
   docker build -t deepprune .
   ```

2. **Run the application:**
   ```bash
   docker run -p 8501:8501 deepprune
   ```

3. **Access the application:**
   - Open your web browser
   - Go to: **http://localhost:8501**

4. **Stop the application:**
   - Press `Ctrl+C` in the terminal

### Run in Background (Detached Mode)

If you want to run the container in the background:

```bash
# Start the container
docker run -d -p 8501:8501 --name deepprune-app deepprune

# View logs
docker logs -f deepprune-app

# Stop the container
docker stop deepprune-app

# Remove the container
docker rm deepprune-app
```

### Docker Troubleshooting

**Port Already in Use:**
If port 8501 is already in use, change it to another port:
```bash
docker run -p 8502:8501 deepprune
```
Then access the app at `http://localhost:8502`

**Rebuild from Scratch:**
```bash
docker build --no-cache -t deepprune .
docker run -p 8501:8501 deepprune
```

**Remove All Stopped Containers:**
```bash
docker container prune
```

**Remove Image:**
```bash
docker rmi deepprune
```

---
