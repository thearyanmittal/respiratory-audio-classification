# Detecting Disease from Respiratory Audio

#### Notes to Teammates:
Download dataset from Kaggle into this repo and rename the downloaded folder ``data/`` for the preprocessing notebook to load it properly.

Instructions for the environment setup (can do on terminal):
1. run “python3 -m venv env” in your directory to create the environment
2. run “source env/bin/activate” to activate it
3. run “pip install -r requirements.txt”

Filetree for midterm:


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

    preprocessing.ipynb: Contains all the preprocessing and model code (for now)
    README.md
    requirements.txt: for venv setup
