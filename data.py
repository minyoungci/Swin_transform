import pandas as pd

root_dir = '/home/minyoungxi/MINYOUNGXI/Kaggle/PetFinder - Swin Transformer'
train_dir = '/home/minyoungxi/MINYOUNGXI/Kaggle/PetFinder - Swin Transformer/train'
test_dir = '/home/minyoungxi/MINYOUNGXI/Kaggle/PetFinder - Swin Transformer/test'

def get_train_file_path(id):
    return f"{train_dir}/{id}.jpg"

df = pd.read_csv(f"{root_dir}/train.csv")
df['file_path'] = df['Id'].apply(get_train_file_path)

class PawpularityDataset(Dataset):
    def __init__(self, root_dir, df, transforms=None):
        self.root_dir = root_dir
        self.df = df
        self.file_names = df['file_path'].values
        self.targets = df['Pawpularity'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return img, target