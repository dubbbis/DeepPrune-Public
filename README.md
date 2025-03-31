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