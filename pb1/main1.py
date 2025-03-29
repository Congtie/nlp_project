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
        """Incarca datele de antrenare din CSV si extrage template-uri pentru rata"""
        self.train_data = pd.read_csv(self.train_csv_path)
        
        # Extragere template pentru rata din imaginea de referinta rata-nitro.png
        template_path = os.path.join(os.path.dirname(self.train_csv_path), "rata-nitro.png")
        if os.path.exists(template_path):
            template = cv2.imread(template_path)
            self.duck_templates.append(template)
            
            # Creeaza masca pentru pixelii ratei
            hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            
            # Vom presupune ca rata are o anumita culoare, adaptati aceste valori in functie de cum arata rata
            lower_color = np.array([20, 100, 100])
            upper_color = np.array([30, 255, 255])
            
            mask = cv2.inRange(hsv, lower_color, upper_color)
            self.duck_masks.append(mask)
        else:
            print(f"Atentie: Nu s-a gasit fisierul template {template_path}")
            
        # Extragere template-uri de rata din imaginile de antrenare
        for _, row in self.train_data.iterrows():
            if row['DuckOrNoDuck'] == 1:
                img_path = os.path.join(self.train_img_dir, f"{row['DatapointID']}.png")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    
                    # Extrage bounding box
                    x1, y1, x2, y2 = map(int, row['BoundingBox'].split())
                    
                    # Extrage template-ul ratei din bounding box
                    duck_template = img[y1:y2+1, x1:x2+1]
                    
                    self.duck_templates.append(duck_template)
                
    def template_matching(self, img):
        """Foloseste template matching pentru a detecta rata"""
        best_score = 0
        best_result = None
        
        for template in self.duck_templates:
            # Verifica daca template-ul nu e None si are o dimensiune valida
            if template is None or template.shape[0] <= 0 or template.shape[1] <= 0:
                continue
                
            # Redimensioneaza template-ul in diferite marimi pentru a gestiona scale
            for scale in np.linspace(0.5, 1.5, 5):
                resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                
                if resized_template.shape[0] > img.shape[0] or resized_template.shape[1] > img.shape[1]:
                    continue
                
                # Foloseste template matching
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
        """Foloseste segmentare pe baza de culoare pentru a detecta rata"""
        # Vom presupune ca rata are o anumita culoare, adaptati aceste valori
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Culorile pentru rata Nitro - acestea trebuie ajustate in functie de cum arata rata
        lower_color = np.array([20, 100, 100])  # HSV pentru galben-portocaliu
        upper_color = np.array([30, 255, 255])
        
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Aplica operatii morfologice pentru a imbunatati masca
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Gaseste contururi
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Ia cel mai mare contur (presupunem ca e rata)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Verifica daca conturul e suficient de mare pentru a fi o rata
        if cv2.contourArea(largest_contour) < 100:
            return None
        
        # Obtine bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        pixel_count = cv2.countNonZero(mask[y:y+h, x:x+w])
        
        return {
            'mask': mask,
            'bbox': (x, y, x + w - 1, y + h - 1),
            'pixel_count': pixel_count
        }
    
    def predict(self, img_path):
        """Prezice daca exista rata in imagine si calculeaza bounding box si numar de pixeli"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"Eroare: Nu s-a putut citi imaginea {img_path}")
            return 0, 0, "0 0 0 0"
        
        # Incercam template matching
        template_result = self.template_matching(img)
        
        # Incercam segmentare pe baza de culoare
        color_result = self.color_segmentation(img)
        
        # Decidem in functie de scorurile obtinute
        if template_result and template_result['score'] > self.threshold:
            x1, y1, x2, y2 = template_result['bbox']
            
            # Folosim masca template-ului pentru a estima numarul de pixeli
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
        """Proceseaza setul de date de test si genereaza fisierul CSV de output"""
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
        
        # Creeaza DataFrame si salveaza ca CSV
        output_df = pd.DataFrame(results)
        
        # Sorteaza dupa DatapointID
        output_df['DatapointID'] = output_df['DatapointID'].astype(int)
        output_df = output_df.sort_values(by='DatapointID')
        output_df['DatapointID'] = output_df['DatapointID'].astype(str)
        
        output_df.to_csv('output.csv', index=False)
        print("Rezultatele au fost salvate in output.csv")

# Functia principala
def main():
    # Defineste caile catre directoare si fisiere conform structurii tale de foldere
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Obtine directorul curent al scriptului
    
    # Corecteaza caile in functie de structura actuala
    train_dataset_dir = os.path.join(base_dir, 'train_dataset')
    train_csv_path = os.path.join(train_dataset_dir, 'train-data.csv')
    test_dataset_dir = os.path.join(base_dir, 'test_dataset')
    
    # Afiseaza caile pentru verificare
    print(f"Cale director de antrenare: {train_dataset_dir}")
    print(f"Cale fisier CSV: {train_csv_path}")
    print(f"Cale director de test: {test_dataset_dir}")
    
    # Verifica daca directoarele si fisierele exista
    if not os.path.exists(train_dataset_dir):
        print(f"Eroare: Directorul {train_dataset_dir} nu exista.")
        return
    
    if not os.path.exists(test_dataset_dir):
        print(f"Eroare: Directorul {test_dataset_dir} nu exista.")
        return
    
    if not os.path.exists(train_csv_path):
        print(f"Eroare: Fisierul {train_csv_path} nu exista.")
        return
    
    # Initializeaza si ruleaza detector-ul de rate
    detector = DuckDetector(train_dataset_dir, train_csv_path, test_dataset_dir)
    detector.load_train_data()
    detector.process_test_data()
    
if __name__ == "__main__":
    main()