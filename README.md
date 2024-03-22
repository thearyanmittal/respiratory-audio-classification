# Detecting Disease from Respiratory Audio

#### Notes to Teammates:
Download dataset from Kaggle into this repo and rename the downloaded folder ``data/`` for the preprocessing notebook to load it properly.

Instructions for the environment setup (can do on terminal):
1. run “python3 -m venv env” in your directory to create the environment
2. run “source env/bin/activate” to activate it
3. run “pip install -r requirements.txt”

Now your filetree should look like the following before you run anything in preprocessing 
(if names of folders are different, please RENAME to match format below)


NAME_OF_REPO_DIRECTORY(Locally, up to you)
    data
        Respiratory_Sound_Database
            audio_and_txt_files
                Contains 1840 files (920 .txt, 920 .wav)
            filename_differences.txt
            filename_format.txt
            patient_diagnosis.csv
        demographic_info.txt
    env
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
    
    preprocessing.ipynb
    README.md
    requirements.txt
