# preprocessing.py
import os
import shutil
from sklearn.model_selection import train_test_split

def organize_dataset(source_dir, output_dir, test_size=0.2, val_size=0.1):
    """
    Organisir dataset menjadi train/val/test split
    """
    print(f"📂 Memproses dataset dari: {source_dir}")
    
    # Cek apakah folder source ada
    if not os.path.exists(source_dir):
        print(f"❌ Error: Folder {source_dir} tidak ditemukan!")
        return
    
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    if len(classes) == 0:
        print(f"❌ Error: Tidak ada folder kelas di {source_dir}")
        return
    
    print(f"✅ Ditemukan {len(classes)} kelas makanan")
    
    # Buat folder output
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    total_images = 0
    
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        
        # Ambil semua file gambar
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(images) == 0:
            print(f"⚠️  Warning: Tidak ada gambar di {class_name}")
            continue
        
        print(f"   📸 {class_name}: {len(images)} gambar")
        total_images += len(images)
        
        # Split data
        train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)
        train_imgs, val_imgs = train_test_split(train_imgs, test_size=val_size/(1-test_size), random_state=42)
        
        # Copy files ke folder yang sesuai
        for split, img_list in [('train', train_imgs), ('validation', val_imgs), ('test', test_imgs)]:
            dest_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(dest_class_dir, exist_ok=True)
            
            for img in img_list:
                src = os.path.join(class_path, img)
                dst = os.path.join(dest_class_dir, img)
                shutil.copy2(src, dst)
    
    print(f"\n✅ Selesai! Total {total_images} gambar diproses")
    print(f"   📁 Train: ~{int(total_images * 0.7)} gambar")
    print(f"   📁 Validation: ~{int(total_images * 0.1)} gambar")
    print(f"   📁 Test: ~{int(total_images * 0.2)} gambar")

if __name__ == '__main__':
    organize_dataset('data/raw', 'data/processed')