# How to run and deploy this project

## A. Run on your own PC (5 minutes)

1. Install Python 3.10, 3.11 or 3.12 (python.org). Tick "Add Python to PATH" on Windows.
2. Open a terminal inside this folder and run:
   ```
   python -m venv venv
   venv\Scripts\activate            (Mac/Linux:  source venv/bin/activate)
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Your browser opens the dashboard. Use the left menu. "Live Demo" lets you upload a retina photo.

Everything in the dashboard works without the DRIVE dataset. You only need the dataset (put it in `data/DRIVE/`, see below) if you want to retrain or regenerate figures.

## B. Deploy on Streamlit Community Cloud (free)

### Step 1: put the project on GitHub
1. Create a free account at github.com and click New repository. Name it `retinal-vessel-segmentation`. Choose Public (the free Streamlit plan needs a public repo, or give Streamlit access to a private one).
2. In a terminal inside this folder:
   ```
   git init
   git add .
   git commit -m "Retinal vessel segmentation project"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/retinal-vessel-segmentation.git
   git push -u origin main
   ```
   The `.gitignore` already keeps the dataset and the big training checkpoints out of GitHub. The small model file `models/final_model.weights.h5` (8 MB) IS included. The app needs it.
3. Check on github.com that these exist: `app.py`, `requirements.txt`, `models/final_model.weights.h5`, `results/gallery/`.

### Step 2: deploy
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click Create app, then "Deploy a public app from GitHub".
3. Fill in: Repository `YOUR-USERNAME/retinal-vessel-segmentation`, Branch `main`, Main file path `app.py`.
4. Click Advanced settings and choose Python 3.12 (or 3.11).
5. Click Deploy. The first build takes about 5 to 10 minutes because TensorFlow is large. You will get a public link like `https://your-app-name.streamlit.app`.

### If something goes wrong
- "Resource limits" or the app restarts when you use Live Demo: the free tier has about 1 GB RAM and TensorFlow is heavy. Fix: open `requirements.txt` and delete the two lines `tensorflow-cpu...` and `keras...`, then push again. The whole dashboard keeps working. Only Live Demo is replaced by a message that TensorFlow is not installed.
- ModuleNotFoundError: add the missing package name to `requirements.txt` and push.
- Build fails on the Python version: choose 3.11 or 3.12 in Advanced settings (Manage app, then Settings).
- Changes you push to GitHub update the app automatically.

## C. Optional: retrain everything yourself

1. Unzip the DRIVE dataset so the folders look like this:
   ```
   data/DRIVE/training/images        data/DRIVE/training/1st_manual     data/DRIVE/training/mask
   data/DRIVE/test/images            data/DRIVE/test/mask
   ```
2. Run in this order (about 1.5 hours on a normal CPU):
   ```
   python -m src.eda
   python -m src.run_experiments
   python -m src.evaluate select
   delete the file experiments/TEST_EVALUATED.lock   (so the final test can run again)
   python -m src.evaluate test
   python -m src.frangi_baseline
   python -m src.make_tables
   python -m src.make_gallery
   ```
   Do not push the dataset to GitHub.
