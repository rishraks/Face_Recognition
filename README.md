# Face Recognition Project

This project is a Python-based real-time Face Recognition system that uses OpenCV, MediaPipe, and a trained machine learning model to detect and recognize faces. The model identifies individuals from live camera feeds with high accuracy by leveraging facial landmarks and bounding boxes.

## Features
- Real-time face detection using MediaPipe Face Detection.
- Face landmark mapping with MediaPipe Face Mesh.
- Accurate face recognition powered by a trained machine learning model.
- Easily extensible for adding new individuals or improving recognition accuracy.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/face-recognition-project.git
   cd face-recognition-project
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the Trained Model:**
   Place your trained model file (`model.p`) in the project directory.

5. **Organize Data for Training (Optional):**
   - Create a folder `Data_Collection` with subfolders named after each person's name.
   - Place corresponding images in each folder.

## Usage

### Real-Time Face Recognition
Run the following command to start the application:
```bash
python testing.py
```

- Press `Q` to quit the application.

### Training a New Model
To train a new model with your dataset:
1. Ensure the dataset is organized in the `Data_Collection` folder.
2. Run the training script:
   ```bash
   python training.py
   ```
3. Save the generated model as `model.p`.

## Requirements
- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn

## File Structure
```
face-recognition-project/
├── Data_Collection/        # Dataset folder with subfolders for each person
├── model.p                # Pre-trained model
├── testing.py             # Script for real-time face recognition
├── training.py            # Script for training the face recognition model
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Future Enhancements
- Add support for multiple cameras.
- Improve recognition accuracy with more training data.
- Implement GUI for user-friendly interaction.

## Acknowledgements
This project leverages the following libraries and tools:
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [Scikit-learn](https://scikit-learn.org/)

## License
This project is licensed under the MIT License. See `LICENSE` for more details.
