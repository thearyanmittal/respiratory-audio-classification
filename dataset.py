# code taken from PyTorch documentation (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import io
from torchvision.transforms import Compose, Normalize, ToTensor

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def get_mean_and_std(dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for sample in dataloader:
        for i in range(3):
            mean[i] += sample['image'][:,i,:,:].mean()
            std[i] += sample['image'][:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

class BreathDataset(Dataset):
    def __init__(self, train, transform):
        self.transform = transform
        self.folds_file = 'data/official_split.txt'
        self.samples_in_fold = set()
        with open(self.folds_file) as file:
            lines = file.readlines()
            for line in lines:
                spl = line.split('\t')
                if train and 'train' in spl[1]:
                    self.samples_in_fold.add(spl[0])
                elif not train and 'test' in spl[1]:
                    self.samples_in_fold.add(spl[0])
        
        df = pd.read_csv('data/data_complete.csv')
        df = df[[any(s in x for s in self.samples_in_fold) for x in df.spectrogram_file]]
        df['spectrogram_file'] = df['spectrogram_file'].apply(lambda x: x.replace('spectrograms', 'normalized_spectrograms'))
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        disease_dict = {'COPD': 0, 'Healthy': 1, 'Pneumonia': 2, 'URTI': 3, 'Bronchiolitis': 4, 'Bronchiectasis': 5, 'LRTI': 6, 'Asthma': 7}
        sex_dict = {'M': 0, 'F': 1}
        chest_location_dict = {'Ar': 0, 'Al': 1, 'Pl': 2, 'Pr': 3, 'Tc': 4, 'Lr': 5, 'Ll': 6}
        equipment_dict = {'AKGC417L': 0, 'Meditron': 1, 'LittC2SE': 2, 'Litt3200': 3}
        sound_dict = {(0,0): 0, (1,0): 1, (0,1): 2, (1,1): 3}
    
        sample = self.df.iloc[idx]
        image = io.read_image(sample['spectrogram_file'])
        wheeze_crackle = (sample['wheezes'], sample['crackles'])
        disease = disease_dict[sample['disease']]
        age = sample['age']
        sex = sex_dict[sample['sex']] if sample['sex'] in sex_dict.keys() else 2
        bmi = sample['bmi']
        chest_location = chest_location_dict[sample['chest_location']]
        recording_equipment = equipment_dict[sample['recording_equipment']]

        if self.transform:
            mean, std = get_mean_and_std(self)
            image = Compose([ToTensor(), Normalize(mean, std)])(image)

        return {'image': image,
                'sound': sound_dict[wheeze_crackle],
                'disease': disease,
                'age': age,
                'sex': sex,
                'bmi': bmi,
                'chest_location': chest_location,
                'recording_equipment': recording_equipment}