import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans

class DuckDetector:
    def __init__(self, train_img_dir, train_csv_path, test_img_dir):
        self.train_img_dir = train_img_dir
        self.train_csv_path = train_csv_path
        self.test_img_dir = test_img_dir
        self.train_data = None
        self.duck_templates = []
        self.duck_masks = []
        self.duck_colors = []
        self.duck_features = []
        self.template_threshold = 0.6
        self.color_thresholds = []
        
    def load_train_data(self):
        """Incarca datele de antrenare si extrage template-uri si caracteristici pentru rata"""
        self.train_data = pd.read_csv(self.train_csv_path)
        
        # Extrage template pentru rata din imaginea de referinta rata-nitro.png daca exista
        template_path = os.path.join(os.path.dirname(self.train_csv_path), "rata-nitro.png")
        if os.path.exists(template_path):
            template = cv2.imread(template_path)
            if template is not None:
                self.duck_templates.append(template)
                self._extract_color_features(template)
        
        # Extrage template-uri de rata din imaginile de antrenare
        positive_samples = self.train_data[self.train_data['DuckOrNoDuck'] == 1]
        for _, row in positive_samples.iterrows():
            img_path = os.path.join(self.train_img_dir, f"{row['DatapointID']}.png")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Extrage bounding box
                    x1, y1, x2, y2 = map(int, row['BoundingBox'].split())
                    
                    # Extrage template-ul ratei din bounding box
                    duck_template = img[y1:y2+1, x1:x2+1]
                    
                    # Verifica daca template-ul are o dimensiune rezonabila
                    if duck_template.shape[0] > 10 and duck_template.shape[1] > 10:
                        self.duck_templates.append(duck_template)
                        self._extract_color_features(duck_template)
                        
                        # Creeaza masca pentru pixelii ratei
                        mask = self._create_duck_mask(duck_template)
                        if mask is not None and np.sum(mask) > 0:
                            self.duck_masks.append(mask)
        
        print(f"Incarcate {len(self.duck_templates)} template-uri de rata")
        print(f"Extrase {len(self.duck_colors)} seturi de caracteristici de culoare")
        
        # Invata intervalele de culoare optime folosind datele de antrenare
        self._learn_color_thresholds()
                
    def _extract_color_features(self, template):
        """Extrage caracteristicile de culoare din template folosind clustering"""
        # Converteste imaginea in spatiul de culoare HSV
        hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        
        # Redimensioneaza imaginea pentru eficienta
        pixels = hsv.reshape((-1, 3))
        
        # Aplica KMeans pentru a gasi culorile predominante
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Obtine centrele clusterelor (culorile predominante)
        colors = kmeans.cluster_centers_
        
        # Calculeaza proportia fiecarei culori in imagine
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        # Adauga caracteristicile de culoare in lista
        self.duck_colors.append(colors)
        
        # Adauga informatii despre distributia culorilor
        color_info = {
            'colors': colors,
            'proportions': counts / len(labels)
        }
        self.duck_features.append(color_info)
    
    def _learn_color_thresholds(self):
        """Invata intervalele de culoare pentru detectia ratei din template-uri"""
        if not self.duck_colors:
            return
            
        # Colecteaza toate culorile predominante din toate template-urile
        all_colors = np.vstack([colors for colors in self.duck_colors])
        
        # Gaseste intervalul minim si maxim pentru fiecare canal H, S, V
        h_min, h_max = np.min(all_colors[:, 0]), np.max(all_colors[:, 0])
        s_min, s_max = np.min(all_colors[:, 1]), np.max(all_colors[:, 1])
        v_min, v_max = np.min(all_colors[:, 2]), np.max(all_colors[:, 2])
        
        # Adauga marje pentru robustete
        h_margin = 10
        s_margin = 30
        v_margin = 30
        
        # Defineste mai multe intervale de culoare pentru a creste robustetea
        # Intervalul principal bazat pe toti template-ii
        self.color_thresholds.append((
            np.array([max(0, h_min - h_margin), max(0, s_min - s_margin), max(0, v_min - v_margin)]),
            np.array([min(180, h_max + h_margin), min(255, s_max + s_margin), min(255, v_max + v_margin)])
        ))
        
        # Intervalul pentru galben-portocaliu (rata Nitro)
        self.color_thresholds.append((
            np.array([15, 100, 100]),
            np.array([35, 255, 255])
        ))
        
        # Intervalul pentru maro (pentru rata Nitro in conditii de luminozitate redusa)
        self.color_thresholds.append((
            np.array([10, 70, 70]),
            np.array([30, 255, 200])
        ))
        
        print(f"Intervale de culoare invatate: {len(self.color_thresholds)}")
    
    def _create_duck_mask(self, img):
        """Creeaza o masca pentru rata folosind informatiile de culoare invatate"""
        if not img.size or img is None:
            return None
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Initializeaza masca goala
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        # Aplica toate intervalele de culoare si combina rezultatele
        for lower_color, upper_color in self.color_thresholds:
            mask = cv2.inRange(hsv, lower_color, upper_color)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Aplica operatii morfologice pentru a imbunatati masca
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        return combined_mask
    
    def template_matching(self, img):
        """Foloseste template matching pentru a detecta rata"""
        best_score = 0
        best_result = None
        
        for template in self.duck_templates:
            # Verifica daca template-ul are dimensiuni valide
            if template is None or template.shape[0] <= 0 or template.shape[1] <= 0:
                continue
                
            # Foloseste mai multe scale pentru a gestiona rata de dimensiuni diferite
            for scale in np.linspace(0.4, 1.6, 6):
                # Redimensioneaza template-ul
                width = int(template.shape[1] * scale)
                height = int(template.shape[0] * scale)
                
                # Verifica daca dimensiunile sunt valide
                if width < 10 or height < 10 or width > img.shape[1] or height > img.shape[0]:
                    continue
                
                resized_template = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
                
                # Foloseste template matching
                result = cv2.matchTemplate(img, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    # Creeaza masca pentru template-ul redimensionat
                    mask = self._create_duck_mask(resized_template)
                    pixel_count = cv2.countNonZero(mask) if mask is not None else 0
                    
                    best_result = {
                        'score': max_val,
                        'template': resized_template,
                        'bbox': (max_loc[0], max_loc[1], max_loc[0] + width - 1, max_loc[1] + height - 1),
                        'mask': mask,
                        'pixel_count': pixel_count
                    }
        
        return best_result
    
    def color_segmentation(self, img):
        """Foloseste segmentare pe baza de culoare pentru a detecta rata"""
        # Initializeaza masca goala
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        # Converteste imaginea in HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Aplica toate intervalele de culoare si combina rezultatele
        for lower_color, upper_color in self.color_thresholds:
            mask = cv2.inRange(hsv, lower_color, upper_color)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Aplica operatii morfologice pentru a imbunatati masca
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Gaseste contururi
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filtreaza contururile dupa arie pentru a elimina zgomotul
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
        
        if not valid_contours:
            return None
        
        # Construieste un scor pentru fiecare contur si alege cel mai probabil sa fie rata
        best_contour = None
        best_score = 0
        
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculeaza scorul bazat pe aria conturului si aspectul raportului formei
            area = cv2.contourArea(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Rata tinde sa aiba un raport de aspect specific
            ratio_score = 1.0 - min(abs(aspect_ratio - 1.0), 1.0)
            
            # Scorul final combina aria si raportul de aspect
            score = area * ratio_score
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is None:
            return None
        
        # Obtine bounding box pentru cel mai bun contur
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Extinde usor bounding box-ul pentru a include toate partile ratei
        x = max(0, x - 2)
        y = max(0, y - 2)
        w = min(img.shape[1] - x, w + 4)
        h = min(img.shape[0] - y, h + 4)
        
        # Numara pixelii din masca in bounding box
        pixel_count = cv2.countNonZero(combined_mask[y:y+h, x:x+w])
        
        return {
            'mask': combined_mask,
            'bbox': (x, y, x + w - 1, y + h - 1),
            'pixel_count': pixel_count,
            'score': best_score
        }
    
    def refine_duck_detection(self, img, bbox, mask):
        """Rafineaza detectia ratei pentru a imbunatati precizia bounding box-ului si numarul de pixeli"""
        x1, y1, x2, y2 = bbox
        
        # Extrage regiunea imaginii care contine rata
        roi = img[y1:y2+1, x1:x2+1]
        
        # Aplica segmentare pe baza de culoare mai precisa in ROI
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Foloseste toate pragurile de culoare pentru a crea o masca combinata
        refined_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        
        for lower_color, upper_color in self.color_thresholds:
            color_mask = cv2.inRange(hsv_roi, lower_color, upper_color)
            refined_mask = cv2.bitwise_or(refined_mask, color_mask)
        
        # Aplica operatii morfologice pentru a imbunatati masca
        kernel = np.ones((3, 3), np.uint8)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Gaseste contururi in masca rafinata
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return bbox, cv2.countNonZero(refined_mask)
        
        # Combina toate contururile pentru a obtine un bounding box precis
        all_points = np.vstack([cnt for cnt in contours])
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Actualizeaza bounding box-ul in coordonatele imaginii originale
        refined_bbox = (x1 + x, y1 + y, x1 + x + w - 1, y1 + y + h - 1)
        
        # Numara pixelii din masca rafinata
        pixel_count = cv2.countNonZero(refined_mask)
        
        return refined_bbox, pixel_count
    
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
        
        # Decidem in functie de rezultatele obtinute
        if template_result and template_result['score'] > self.template_threshold:
            bbox = template_result['bbox']
            mask = template_result['mask']
            
            # Rafinam rezultatul pentru o mai buna precizie
            if mask is not None:
                refined_bbox, pixel_count = self.refine_duck_detection(img, bbox, mask)
                x1, y1, x2, y2 = refined_bbox
            else:
                x1, y1, x2, y2 = bbox
                pixel_count = template_result['pixel_count']
            
            return 1, pixel_count, f"{x1} {y1} {x2} {y2}"
        
        elif color_result:
            bbox = color_result['bbox']
            mask = color_result['mask']
            
            # Rafinam rezultatul pentru o mai buna precizie
            refined_bbox, pixel_count = self.refine_duck_detection(img, bbox, mask)
            x1, y1, x2, y2 = refined_bbox
            
            return 1, pixel_count, f"{x1} {y1} {x2} {y2}"
        
        else:
            return 0, 0, "0 0 0 0"
    
    def process_test_data(self):
        """Proceseaza setul de date de test si genereaza fisierul CSV de output"""
        results = []
        
        test_files = [f for f in os.listdir(self.test_img_dir) if f.endswith('.png')]
        
        # Sorteaza fisierele dupa ID pentru a asigura ordinea corecta
        test_files.sort(key=lambda x: int(x.split('.')[0]))
        
        total_files = len(test_files)
        for i, file in enumerate(test_files):
            datapoint_id = file.split('.')[0]
            img_path = os.path.join(self.test_img_dir, file)
            
            # Afiseaza progresul
            if (i + 1) % 10 == 0 or i == 0 or i == total_files - 1:
                print(f"Procesare imagine {i+1}/{total_files}: {file}")
            
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