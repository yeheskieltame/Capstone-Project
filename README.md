# C242-PS377 
## Waste Classification with MobileNetV2

This project aims to classify waste types into several categories (cardboard, glass, metal, paper, plastic, trash, and organic) using a pre-trained MobileNetV2 model. The model can be used to help in automatic waste sorting, which can be beneficial for recycling and more efficient waste management.
## Contributors
 - M239b4ky4526 - Yeheskiel Yunus Tame - Universitas Kristen Duta Wacana - Machine Learning
 - M239b4ky1557 - Frederik Samra Sarongallo - Universitas Kristen Duta Wacana - Machine Learning
 - M314b4ky2563 - Mohammad Baharudin Yusuf - Universitas Singaperbangsa Karawang -  Machine Learning 
## Features

* **Multi-class Classification:** The model can classify waste into 8 different categories.
* **High Accuracy:** The model is trained on a large dataset and achieves high accuracy on validation data.
* **Efficiency:** Using MobileNetV2, this model is relatively lightweight and fast, making it suitable for implementation on resource-limited devices.
* **Data Augmentation:** Employs data augmentation techniques to increase training data variation and prevent overfitting.
* **Transfer Learning:** Leverages transfer learning from MobileNetV2 pre-trained on the ImageNet dataset, allowing the model to learn faster and more effectively.

## How to Use

### 1. Prepare Dataset

* **Location:** Place your waste images in the `Capstone-Project/Dataset` folder
* **Folder Structure:** Each waste category should have its own folder
* **Supported Categories:**
  - Cardboard
  - Glass
  - Metal
  - Paper
  - Plastic
  - Trash
  - Organic

### 2. Run the Notebook

* **Platform:** Use Google Colab or Jupyter Notebook
* **File:** Open `model_waste_classification.ipynb`
* **Process:**
  1. Load required libraries
  2. Prepare dataset
  3. Train the model
  4. Evaluate the model
* **Run all cells sequentially**

### 3. Model Testing

* **Prediction Function:** Use `predict_image(image_path)`
* **Parameter:** Input the path of the image to be tested
* **Usage Example:**
  ```python
  # Example of calling the prediction function
  result = predict_image('path/to/your/image.jpg')
  print(result)
