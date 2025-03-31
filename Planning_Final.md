# DeepPrune: Optimized Workflow  

## Week 1: Data Processing & Image Selection  

### Day 1: Dataset Preparation  
- Load and extract EXIF metadata from drone images.  
- Store metadata in a structured CSV dataset.  
- Verify data integrity and handle missing values.  

### Day 2: Data Exploration & Visualization  
- Generate statistical summaries of altitude, GPS, and orientation.  
- Visualize geospatial heatmaps of image locations.  
- Detect outliers and potential inconsistencies.  

### Day 3: Image Overlap Estimation (Step 1)  
- Compute pairwise distances between images using latitude, longitude, and altitude.  
- Set a minimum distance threshold to filter out redundant images.  
- Identify potential duplicate images based on GPS proximity.  

### Day 4: Image Orientation Filtering (Step 2)  
- Compare orientation angles between nearby images.  
- Remove images with similar orientation (<10° difference) to avoid redundant views.  
- Generate a refined subset of non-redundant images.  

### Day 5: Feature Extraction with CNN (Step 3)  
- Extract deep visual features from selected images using ResNet50.  
- Compute Cosine Similarity to measure visual redundancy.  
- Cluster images using K-Means to group visually similar images.  

## Week 2: Model Training & Final Optimization  

### Day 6: Representative Image Selection  
- Select the most informative image from each cluster.  
- Optimize the subset size while maintaining scene coverage.  

### Day 7: Dataset Finalization  
- Store the final selected images in a structured dataset.  
- Perform final data integrity checks.  

### Day 8-9: CNN-Based Model Training  
- Train a CNN-based model to predict optimal image selection from raw datasets.  
- Evaluate model performance using the manually optimized dataset.  

### Day 10: Final Testing & Optimization  
- Test the CNN model on a new dataset.  
- Fine-tune model parameters for improved performance.  
- Validate results and ensure high-quality selection.  

### Day 11-12: Presentation & Documentation  
- Prepare presentation slides explaining the methodology.  
- Document the workflow, challenges, and results.  
- Finalize the GitHub repository with the full project.  
