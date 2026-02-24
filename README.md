<h1 align="center">🕵️ Deepfake Detection using Deep Learning</h1>



<p align="center">
  <b>AI-based system for detecting manipulated facial images using CNN architectures</b>
</p>


<hr>

<h2>📌 Project Overview</h2>
<p>
This project implements a deep learning-based approach for detecting deepfake images.
We fine-tune a pretrained <b>ResNet18</b> model to classify images as <b>Real</b> or <b>Fake</b>.
The system also evaluates robustness under resolution degradation and noise conditions.
</p>

<hr>

<h2>🧠 Model Architecture</h2>
<ul>
  <li>Pretrained ResNet18 (Transfer Learning)</li>
  <li>Binary classification (Real vs Fake)</li>
  <li>Fine-tuning last convolutional layers</li>
  <li>Cross-Entropy Loss</li>
  <li>Adam Optimizer</li>
</ul>

<hr>

<h2>📊 Performance</h2>
<table>
  <tr>
    <th>Metric</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Validation Accuracy</td>
    <td>83.5%</td>
  </tr>
  <tr>
    <td>Training Loss (Final)</td>
    <td>0.0277</td>
  </tr>
</table>

<hr>

<h2>🔬 Robustness Evaluation</h2>
<p>
The model was tested under different degradation conditions:
</p>

<ul>
  <li>Low Resolution (128x128)</li>
  <li>Image Noise Injection</li>
  <li>Compression Simulation</li>
</ul>

<p>
Results show a moderate drop in performance under heavy degradation,
highlighting sensitivity to compression artifacts.
</p>

<hr>

<h2>📁 Project Structure</h2>

<pre>
📦 Deepfake-Detection
 ┣ 📂 data
 ┣ 📂 notebooks
 ┣ 📂 models
 ┣ 📄 train.py
 ┣ 📄 evaluate.py
 ┗ 📄 README.md
</pre>

<hr>

<h2>🚀 How to Run</h2>

<pre>
pip install -r requirements.txt
python train.py
python evaluate.py
</pre>

<hr>

<h2>🛠 Technologies Used</h2>
<ul>
  <li>Python 3.11</li>
  <li>PyTorch</li>
  <li>Torchvision</li>
  <li>NumPy</li>
  <li>Matplotlib</li>
</ul>

<hr>

<h2>📌 Future Improvements</h2>
<ul>
  <li>Add Grad-CAM visualization</li>
  <li>Integrate video-level temporal analysis</li>
  <li>Improve robustness with advanced augmentation</li>
  <li>Deploy via Streamlit web app</li>
</ul>

<hr>

<h2>👩‍💻 Author</h2>
<p>
Safa — Artificial Intelligence & Cybersecurity Student
</p>

<p align="center">
⭐ If you like this project, give it a star!
</p>
