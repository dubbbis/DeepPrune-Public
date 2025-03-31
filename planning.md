# **DeepPrune Project Workflow**

## **Week 1: MVP Development**
### **Day 1: Data Collection & Preprocessing**
- Organize photogrammetry images into structured folders.
- Extract EXIF metadata (GPS coordinates, altitude, orientation).
- Create an initial dataset (CSV) with relevant image information.

### **Day 2: Data Exploration & Visualization**
- Load and inspect the dataset.
- Handle missing or corrupted data.
- Generate statistical summaries and distributions.
- Visualize geospatial data (GPS heatmaps, altitude histograms).

### **Day 3: Feature Engineering & Selection**
- Extract additional relevant features from EXIF data.
- Implement image overlap estimation.
- Minimum distance between images.
- Orientation angle (to avoid very similar images).
- Overlap threshold based on clustering:
    - K-Means clustering is applied to visual features extracted with CNN (ResNet50) to group images that contain redundant visual content.
    - A representative image is then selected from each cluster, reducing redundancies without losing key information.
- Select optimal features for CNN-based model training.

### **Day 4: Model Development - CNN Optimization**
- Define CNN architecture for image selection.
- Implement model training pipeline.
- Validate with a small dataset.

### **Day 5: Model Evaluation & Refinement**
- Test model performance on validation data.
- Adjust hyperparameters to improve accuracy.
- Implement logging and evaluation metrics.

### **Day 6-7: MVP Completion & API Integration**
- Automate the selection process using the trained CNN.
- Integrate model into an API or pipeline for future automation.
- Finalize MVP with a functional workflow.

---

## **Week 2: Presentation & Optimization**
### **Day 8: Fine-tuning & Optimizations**
- Optimize the image selection process.
- Reduce processing time and memory consumption.
- Conduct further testing to validate improvements.

### **Day 9: Documentation & Report Preparation**
- Write detailed project documentation.
- Describe methodology, dataset, and results.
- Prepare markdown-based README for GitHub.

### **Day 10: Presentation Development**
- Create slides summarizing key findings and model performance.
- Include visuals, graphs, and step-by-step explanations.

### **Day 11: Code Refinement & Edge Case Handling**
- Improve code readability and modularity.
- Handle potential edge cases (corrupted images, missing metadata).

### **Day 12: Rehearsal & Feedback Session**
- Present MVP to team or mentor for feedback.
- Make necessary revisions based on insights.

### **Day 13-14: Final Adjustments & Presentation Delivery**
- Polish slides and script.
- Deliver final presentation.
- Summarize future improvements and next steps.

---


