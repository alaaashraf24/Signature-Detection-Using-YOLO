# YOLOv8 Signature Detection

This project demonstrates how to train a YOLOv8 model for signature object detection using a custom dataset. The code is designed to run efficiently in a Google Colaboratory or Kaggle Notebook environment.

## Project Overview

The goal is to identify and locate signatures within document images. The process involves:
1.  **Dataset Preparation**: Structuring the provided signature dataset into the format required by YOLOv8. This includes converting TIFF images to JPG and transforming absolute bounding box coordinates to YOLO format (normalized center coordinates, width, and height).
2.  **Data Splitting**: Splitting the prepared dataset into training and validation sets.
3.  **Model Training**: Training a YOLOv8 model (specifically YOLOv8n, the nano version, for speed) on the prepared dataset.
4.  **Model Evaluation**: Evaluating the trained model's performance on a test set.
5.  **Inference**: Using the trained model to detect signatures on new, unseen images.

## Requirements

*   Python 3.x
*   Google Colaboratory or Jupyter Notebook environment (preferably with GPU access for faster training)
*   Required Python libraries:                                                                                                                                               - Python 3.8+
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- PyTorch
- OpenCV
- Pandas, NumPy
- Matplotlib

## Setup

1.  **Open the Notebook**: Upload and open the provided Python code in a Google Colaboratory or Kaggle Notebook.
2.  **Dataset**: Ensure your dataset is accessible within the notebook environment. The code assumes the dataset is located at `/kaggle/input/signature-object-detection/SignatureObjectDetection` and `/kaggle/input/testimage/test.tif`, `/kaggle/input/test2image/test2.tif` for testing, modify `BASE_PATH` and test image paths if your dataset location is different.
3.  **Install Dependencies**: The notebook includes the command `!pip install ultralytics` to install the necessary library.

## Code Structure

The notebook is structured into several sections:

1.  **Import Libraries**: Imports all required Python libraries (`os`, `shutil`, `pathlib`, `yaml`, `PIL`, `cv2`, `numpy`, `pandas`, `ultralytics`, `matplotlib`).
2.  **Dataset Paths**: Defines variables for the paths to the original dataset directories and the planned output directories for the YOLO-formatted dataset.
3.  **Helper Functions**: Contains functions to perform specific tasks:
    *   `create_directory_structure()`: Sets up the `train/images`, `train/labels`, `val/images`, `val/labels` directories.
    *   `convert_bbox_to_yolo_format()`: Converts `[x1, y1, x2, y2]` coordinates to YOLO `[x_center, y_center, width, height]` normalized format.
    *   `read_ground_truth_file()`: Reads bounding box coordinates from the ground truth text files.
    *   `process_dataset()`: Iterates through images and ground truth files, converts images to JPG, transforms bounding boxes, and saves them in the YOLO format.
    *   `create_dataset_yaml()`: Generates the `dataset.yaml` configuration file for YOLO training.
    *   `split_dataset_for_validation()`: Moves a portion of the processed training data to the validation directories.
    *   `train_yolo_model()`: Initializes and trains the YOLOv8 model.
    *   `evaluate_model_on_test_set()`: Processes test images and runs inference to demonstrate model predictions.
    *   `visualize_training_results()`: Displays key training plots saved by YOLO.
    *   `infer_and_display()`: Runs inference on a single image path and displays the result with detected bounding boxes.
4.  **Main Execution Flow**: Orchestrates the steps for dataset preparation, training, and evaluation.
5.  **Test The Model On New Images**: Loads the best trained model and runs inference on example test images.

## How to Run

Simply run the notebook cells sequentially in your Colab or Kaggle environment.

The main execution flow cell will perform the following steps:

1.  Create the required directory structure.
2.  Process the training dataset (convert images to JPG, convert annotations to YOLO format).
3.  Split the processed training data into training and validation sets (default 80/20 split).
4.  Create the `dataset.yaml` configuration file.
5.  Train the YOLOv8n model for 100 epochs (default).
6.  Evaluate the model by running inference on the first 5 test images and saving the results.
7.  Visualize training metrics plots.
8.  Print the final performance metrics from the training results CSV.
9.  Load the best trained model weights.
10. Run inference on the specified test images and display the results.

## Output

*   The processed dataset in YOLO format will be created in `/kaggle/working/yolo_dataset`.
*   Training results, including weights, plots, and metrics, will be saved in `/kaggle/working/runs/signature_detectionX` (where X is an incrementing number, likely 2 based on the code).
*   Predicted images from the test set evaluation will be saved in `/kaggle/working/prediction_*.jpg`.
*   Visualizations of training plots will be displayed within the notebook.
*   The final performance metrics (Precision, Recall, mAP) will be printed.
*   Inference results on the example test images with bounding boxes will be displayed.

## Customization

*   **Dataset Paths**: Modify `BASE_PATH`, `TRAIN_IMAGES_PATH`, etc., if your dataset is located elsewhere.
*   **Training Parameters**: Adjust `epochs`, `img_size`, `batch_size` in the `train_yolo_model` function call for different training configurations.
*   **Model Choice**: The code uses `yolov8n.pt`. You can change this to other YOLOv8 models like `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, or `yolov8x.pt` by modifying the `YOLO()` initialization line in `train_yolo_model`. Note that larger models require more resources and training time.
*   **Validation Split**: Change the `train_split_ratio` argument in `split_dataset_for_validation()` to adjust the train/validation split percentage.
*   **Test Evaluation Count**: Modify the `[:5]` slice in the `evaluate_model_on_test_set` function to process more or fewer test images.
*   **Inference Images**: Change the paths provided to the `infer_and_display` function to test on different images.

## Troubleshooting

*   **File Not Found Errors**: Double-check the dataset paths and ensure the notebook environment has access to the files.
*   **CUDA Errors**: Ensure you are using a GPU runtime in Colab/Kaggle if you encounter CUDA out-of-memory or similar errors, or reduce the `batch_size`.
*   **Invalid Coordinate Count**: The `read_ground_truth_file` function expects coordinate values that are integers and groups them into sets of 4 `[x1, y1, x2, y2]`. Ensure your ground truth files follow this format.
*   **YOLO Training Errors**: Consult the Ultralytics YOLO documentation for specific training errors.
