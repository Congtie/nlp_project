import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim

class DuckDetector:
    def __init__(self, train_img_dir, train_csv_path, test_img_dir):
        self.train_img_dir = train_img_dir
        self.train_csv_path = train_csv_path
        self.test_img_dir = test_img_dir
        self.train_data = None
        self.duck_templates = []
        self.duck_masks = []
        self.threshold = 0.7  # Threshold for similarity
        
    def load_train_data(self):
        """Încarcă datele de antrenare din CSV și extrage template-uri pentru rață"""
        self.train_data = pd.read_csv(self.train_csv_path)
        
        # Extragere template pentru rață din imaginea de referință rata-nitro.png
        template_path = os.path.join(os.path.dirname(self.train_csv_path), "rata-nitro.png")
        if os.path.exists(template_path):
            template = cv2.imread(template_path)
            self.duck_templates.append(template)
            
            # Creează mască pentru pixelii raței
            hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            
            # Vom presupune că rața are o anumită culoare, adaptați aceste valori în funcție de cum arată rața
            lower_color = np.array([20, 100, 100])
            upper_color = np.array([30, 255, 255])
            
            mask = cv2.inRange(hsv, lower_color, upper_color)
            self.duck_masks.append(mask)
        else:
            print(f"Atenție: Nu s-a găsit fișierul template {template_path}")
            
        # Extragere template-uri de rață din imaginile de antrenare
        for _, row in self.train_data.iterrows():
            if row['DuckOrNoDuck'] == 1:
                img_path = os.path.join(self.train_img_dir, f"{row['DatapointID']}.png")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    
                    # Extrage bounding box
                    x1, y1, x2, y2 = map(int, row['BoundingBox'].split())
                    
                    # Extrage template-ul raței din bounding box
                    duck_template = img[y1:y2+1, x1:x2+1]
                    
                    self.duck_templates.append(duck_template)
                
    def template_matching(self, img):
        """Folosește template matching pentru a detecta rața"""
        best_score = 0
        best_result = None
        
        for template in self.duck_templates:
            # Verifică dacă template-ul nu e None și are o dimensiune validă
            if template is None or template.shape[0] <= 0 or template.shape[1] <= 0:
                continue
                
            # Redimensionează template-ul în diferite mărimi pentru a gestiona scale
            for scale in np.linspace(0.5, 1.5, 5):
                resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                
                if resized_template.shape[0] > img.shape[0] or resized_template.shape[1] > img.shape[1]:
                    continue
                
                # Folosește template matching
                result = cv2.matchTemplate(img, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    h, w = resized_template.shape[:2]
                    best_result = {
                        'score': max_val,
                        'template': resized_template,
                        'bbox': (max_loc[0], max_loc[1], max_loc[0] + w - 1, max_loc[1] + h - 1)
                    }
        
        return best_result
    
    def color_segmentation(self, img):
        """Folosește segmentare pe bază de culoare pentru a detecta rața"""
        # Vom presupune că rața are o anumită culoare, adaptați aceste valori
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Culorile pentru rața Nitro - acestea trebuie ajustate în funcție de cum arată rața
        lower_color = np.array([20, 100, 100])  # HSV pentru galben-portocaliu
        upper_color = np.array([30, 255, 255])
        
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Aplică operații morfologice pentru a îmbunătăți masca
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Găsește contururi
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Ia cel mai mare contur (presupunem că e rața)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Verifică dacă conturul e suficient de mare pentru a fi o rață
        if cv2.contourArea(largest_contour) < 100:
            return None
        
        # Obține bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        pixel_count = cv2.countNonZero(mask[y:y+h, x:x+w])
        
        return {
            'mask': mask,
            'bbox': (x, y, x + w - 1, y + h - 1),
            'pixel_count': pixel_count
        }
    
    def predict(self, img_path):
        """Prezice dacă există rață în imagine și calculează bounding box și număr de pixeli"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"Eroare: Nu s-a putut citi imaginea {img_path}")
            return 0, 0, "0 0 0 0"
        
        # Încercăm template matching
        template_result = self.template_matching(img)
        
        # Încercăm segmentare pe bază de culoare
        color_result = self.color_segmentation(img)
        
        # Decidem în funcție de scorurile obținute
        if template_result and template_result['score'] > self.threshold:
            x1, y1, x2, y2 = template_result['bbox']
            
            # Folosim masca template-ului pentru a estima numărul de pixeli
            template = template_result['template']
            hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            lower_color = np.array([20, 100, 100])
            upper_color = np.array([30, 255, 255])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            pixel_count = cv2.countNonZero(mask)
            
            return 1, pixel_count, f"{x1} {y1} {x2} {y2}"
        
        elif color_result:
            x1, y1, x2, y2 = color_result['bbox']
            return 1, color_result['pixel_count'], f"{x1} {y1} {x2} {y2}"
        
        else:
            return 0, 0, "0 0 0 0"
    
    def process_test_data(self):
        """Procesează setul de date de test și generează fișierul CSV de output"""
        results = []
        
        test_files = [f for f in os.listdir(self.test_img_dir) if f.endswith('.png')]
        for file in sorted(test_files, key=lambda x: int(x.split('.')[0])):
            datapoint_id = file.split('.')[0]
            img_path = os.path.join(self.test_img_dir, file)
            
            duck_or_no_duck, pixel_count, bounding_box = self.predict(img_path)
            
            results.append({
                'DatapointID': datapoint_id,
                'DuckOrNoDuck': duck_or_no_duck,
                'PixelCount': pixel_count,
                'BoundingBox': bounding_box
            })
        
        # Creează DataFrame și salvează ca CSV
        output_df = pd.DataFrame(results)
        
        # Sortează după DatapointID
        output_df['DatapointID'] = output_df['DatapointID'].astype(int)
        output_df = output_df.sort_values(by='DatapointID')
        output_df['DatapointID'] = output_df['DatapointID'].astype(str)
        
        output_df.to_csv('output.csv', index=False)
        print(f"Rezultatele au fost salvate în output.csv")

# Funcția principală
def main():
    # Definește căile către directoare și fișiere conform structurii tale de foldere
    train_dataset_dir = 'dataset/train_dataset'
    train_csv_path = os.path.join(train_dataset_dir, 'train-data.csv')
    test_dataset_dir = 'dataset/test_dataset/test_dataset'
    
    # Verifică dacă directoarele și fișierele există
    if not os.path.exists(train_dataset_dir):
        print(f"Eroare: Directorul {train_dataset_dir} nu există.")
        return
    
    if not os.path.exists(test_dataset_dir):
        print(f"Eroare: Directorul {test_dataset_dir} nu există.")
        return
    
    if not os.path.exists(train_csv_path):
        print(f"Eroare: Fișierul {train_csv_path} nu există.")
        return
    
    # Inițializează și rulează detector-ul de rațe
    detector = DuckDetector(train_dataset_dir, train_csv_path, test_dataset_dir)
    detector.load_train_data()
    detector.process_test_data()
    
if __name__ == "__main__":
    main()