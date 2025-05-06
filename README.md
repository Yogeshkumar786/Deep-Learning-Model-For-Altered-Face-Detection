# Deep-Learning-Model-For-Altered-Face-Detection
Welcome to our project, Using Deep Learning to Detect Altered Faces! We built an automated deepfake detection system to tackle the growing threat of manipulated videos that blur the line between truth and deception. Using a CNN-LSTM model with an attention mechanism, trained on the FaceForensics++ dataset (~140 videos), our system identifies fake videos with high accuracy while explaining its decisions through visualizations. As beginner AI enthusiasts, we’re thrilled to share this accessible tool, hosted at https://github.com/Raja-Mehdi-Ali-Khan/Using-Deep-Learning-to-Detect-Altered-Faces. Inspired by George E. P. Box’s wisdom, “All models are wrong, some are useful,” our model is a practical step toward safeguarding trust in the digital age.

Introduction
Deepfakes—videos where faces are altered to deceive viewers—are a rising challenge in today’s media landscape. From fake celebrity endorsements to misleading political content, these manipulations undermine trust in what we see. Our project aims to combat this by developing a deep learning system that automatically detects altered faces in videos. Using a CNN-LSTM architecture with an attention mechanism, we process videos from the FaceForensics++ dataset, extracting features like facial landmarks and motion patterns to distinguish real from fake. Our system not only achieves strong performance but also visualizes its focus areas, making it understandable for beginners. This repository contains our Kaggle notebook, outputs, and instructions to replicate or extend our work, inviting others to join us in restoring faith in visual media.

Motivation
As students diving into AI, we were struck by the real-world impact of deepfakes. News outlets, social media, and even personal communications are vulnerable to manipulated videos that can spread misinformation rapidly. We wanted to build a tool that empowers users to verify video authenticity, especially in an era where trust is fragile. Our journey was both challenging and rewarding—learning to wrangle video data, tune a complex model, and visualize results taught us the power of teamwork and deep learning. By sharing this project, we hope to inspire other beginners to tackle pressing problems with AI, creating solutions that are both effective and approachable. Our system, while not perfect, reflects our commitment to making a difference, one video at a time.

Features
Our system includes five key features, detailed in our report:

Comprehensive Feature Extraction: Captures motion patterns, facial landmarks, and deep features to robustly analyze videos.
CNN-LSTM with Attention: Combines convolutional layers for spatial analysis, LSTM for temporal patterns, and attention to prioritize critical frames.
Attention Visualization: Generates heatmaps (e.g., /kaggle/working/attention_weights.png) showing which frames or regions the model focuses on.
Data Augmentation: Applies flips, rotations, and noise to enhance model generalization on a small dataset.
Single-Video Prediction: Enables real/fake classification for new videos with a single inference, ideal for practical use.
Requirements
Software:
Python 3.8 or higher
Libraries: TensorFlow, OpenCV, NumPy, Matplotlib, Pandas
Hardware:
GPU recommended (e.g., Kaggle’s P100 or local NVIDIA GPU with CUDA)
CPU viable but slower for training
Dataset:
FaceForensics++ (~140 videos, available on Kaggle: link-to-kaggle-dataset)
Optional:
Jupyter Notebook for local runs
Kaggle account for cloud execution
Installation
Follow these steps to set up the project:

Clone the Repository:

git clone 
cd Using-Deep-Learning-to-Detect-Altered-Faces
Set Up a Virtual Environment (recommended for local runs):

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install Dependencies:

pip install -r requirements.txt
Or install manually:

pip install tensorflow==2.10.0 opencv-python==4.5.5.64 numpy==1.21.6 matplotlib==3.5.1 pandas==1.3.5
Prepare the Dataset:

Download FaceForensics++ from Kaggle: link-to-kaggle-dataset
For Kaggle: Attach the dataset to your notebook via /kaggle/input/faceforensics.
For local runs: Place videos in ./data/faceforensics/ and update notebook paths accordingly.
Verify Setup:

Ensure GPU availability (run nvidia-smi locally or check Kaggle’s GPU settings).
Test Python imports: python -c "import tensorflow, cv2, numpy, matplotlib, pandas"
Usage
Our project is implemented in a Kaggle notebook. Here’s how to use it:

Run on Kaggle:

Upload deepfake_detection.ipynb to Kaggle.
Attach the FaceForensics++ dataset via /kaggle/input/faceforensics.
Enable GPU in Kaggle’s settings (Session Options > Accelerator > GPU P100).
Run all cells to:
Train the CNN-LSTM model (20 epochs).
Generate plots: /kaggle/working/accuracy_plot.png (accuracy vs. epochs), /kaggle/working/loss_plot.png (loss vs. epochs), /kaggle/working/attention_weights.png (attention heatmap).
Save model weights to /kaggle/working/model_weights.h5.
Download outputs from /kaggle/working for analysis.
Run Locally:

Update dataset paths in the notebook (e.g., ./data/faceforensics).
Launch Jupyter:
jupyter notebook deepfake_detection.ipynb
Run cells to train and generate outputs in ./outputs/.
Ensure GPU drivers and CUDA are installed for faster training.
Test on a New Video:

Place a video (e.g., test_video.mp4) in ./test_videos/ (local) or /kaggle/input/test_videos/ (Kaggle).
Update the notebook’s prediction cell with the video path.
Run to classify as real or fake and generate an attention heatmap (/kaggle/working/test_attention.png).
Example output: “Video is FAKE with 85% confidence.”
View Results:

Check /kaggle/working/accuracy_plot.png for training trends.
Inspect /kaggle/working/attention_weights.png to see which frames the model prioritizes.
Review /kaggle/working/model_metrics.txt for final metrics (e.g., test accuracy).
Results
Our model performed impressively on the FaceForensics++ test set, as detailed in our report:

Test Accuracy: 92.86%, correctly classifying most videos.
Precision: 100%, avoiding false positives.
Recall: 85.71%, catching most fakes.
Validation Loss: 0.1689 after 20 epochs, showing stable learning.
Visualizations:
Accuracy plot (/kaggle/working/accuracy_plot.png): Shows steady improvement to 92.86%.
Loss plot (/kaggle/working/loss_plot.png): Displays decreasing loss curves.
Attention heatmap (/kaggle/working/attention_weights.png): Highlights key frames, making decisions transparent.
These results demonstrate our model’s reliability and interpretability, making it a valuable tool for beginners and researchers combating deepfakes.

Future Work
As Box’s quote suggests, our model is useful but has room to grow. We plan to:

Integrate audio features to detect voice manipulations, complementing visual analysis.
Train on larger, more diverse datasets to improve robustness across video types.
Develop real-time detection for platforms like social media or news outlets.
Enhance attention visualizations with interactive tools for deeper insights.
Optimize the model for CPU-based deployment, increasing accessibility.
These improvements aim to make our system more practical and impactful, addressing current limitations like occasional misclassifications.

Contributing
We welcome contributions to enhance this project! To contribute:

Fork the repository.
Create a branch: git checkout -b feature/your-feature.
Commit changes: git commit -m "Add your feature".
Push to GitHub: git push origin feature/your-feature.
Open a pull request.
For bugs or ideas, open an issue on GitHub. Please follow our code of conduct.

Acknowledgments
FaceForensics++ Team: For providing the dataset that made this project possible.
Kaggle: For free GPU resources and an accessible platform.
Our Team: Raja Mehdi Ali Khan, Yogesh Kumar and Lakshaya Vardhan for designing, coding, and learning as beginner AI enthusiasts.
Instructors and Peers: For guidance and feedback throughout our journey.
For questions or support, open an issue on GitHub or contact yogeshkumarmaliksinghjat200@gmail.com . Join us in using deep learning to detect altered faces and restore trust in what’s real!
