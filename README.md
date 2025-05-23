# Detecting Disease from Respiratory Audio

#### Notes to Teammates:
Download dataset from Kaggle into this repo and rename the downloaded folder ``data/`` for the preprocessing notebook to load it properly.
DO NOT add PyTorch packages to requirements.txt, since installations differ by hardware (freeze before installing Torch)

official split file obtained from official ICBHI challenge site
reason for using bmi over height/weight is for visualization + prevent overfitting of decision tree based models
need grayscale (1 color channel) versions of spectrograms for AST and/or any non-3-channel-pretrained models (from scratch models)

Instructions for the environment setup (can do on terminal):
1. run “python3 -m venv env” in your directory to create the environment
2. run “source env/bin/activate” to activate it
3. run “pip install -r requirements.txt”

Filetree for final: - ipynb files are spread throughout branches


Group23/

    data/: Contains all the data for the project, did not push to Github to avoid large file issues, on local machine
    
        Respiratory_Sound_Database/: Contains the dataset from Kaggle
            audio_and_txt_files/: Contains all the audio files and their corresponding text files
            clips_by_cycle.csv: Contains the cycle number for each clip
            filename_differences.txt: A text file listing 91 names
            filename_format.txt: Explains the file naming format
            patient_diagnosis.csv: Lists the diagnosis for each patient

        downsampled_clips/: Contains the downsampled clips
        normalized_spectograms/: Contains the normalized spectograms
        spectograms/: Contains the mel spectograms
        data_by_cycle_and_demographics.csv: Preprocessing intermediate file
        data_by_cycle.csv: Preprocessing intermediate file
        data_complete.csv: Complete data file
        demographic_info.txt: Intermediate file

    env/: Contains the virtual environment
        bin (folder)
        include (folder)
        lib (folder)
        share (folder)
        pyvenv.cfg

    .gitignore
        Contains the lines below, can just copy paste:
        env/
        .gitignore
        data/
        .DS_STORE

    preprocessing.ipynb: Contains all the preprocessing code and GMM
    CNN.ipynb: contians the CNN
    audio_clustering.ipynb: For DBSCAN on audio
    audio_pca.ipynb: simplifies computations
    README.md
    requirements.txt: for venv setup
