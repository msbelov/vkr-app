import os
import cv2
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
from molscribe import MolScribe
import torch
import numpy as np
from pdf2image import convert_from_path
import tempfile
import time

class ChemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chemical Structure Detector")
        self.root.geometry("800x600")

        self.use_molscribe = tk.BooleanVar(value=False)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.message_queue = queue.Queue()

        self.processing = False
        self.pdf_pages = []

        self.create_widgets()

        self.load_models()
        self.start_time = None

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)

        self.molscribe_check = ttk.Checkbutton(
            settings_frame,
            text="Использовать MolScribe для распознавания структур",
            variable=self.use_molscribe
        )
        self.molscribe_check.pack(anchor=tk.W)

        load_frame = ttk.LabelFrame(main_frame, text="Загрузка данных", padding="10")
        load_frame.pack(fill=tk.X, pady=5)

        self.btn_load = ttk.Button(load_frame, text="Выбрать файл (изображение/PDF)", command=self.load_file)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.file_label = ttk.Label(load_frame, text="Файл не загружен")
        self.file_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.btn_output = ttk.Button(load_frame, text="Выбрать папку для сохранения", command=self.select_output_dir)
        self.btn_output.pack(side=tk.LEFT, padx=5)

        self.output_label = ttk.Label(load_frame, text="Папка не выбрана")
        self.output_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Кнопки управления
        button_frame = ttk.Frame(main_frame, padding="10")
        button_frame.pack(fill=tk.X, pady=5)

        self.btn_process = ttk.Button(button_frame, text="Начать обработку", command=self.start_processing)
        self.btn_process.pack(side=tk.LEFT, padx=5)

        self.btn_clear_log = ttk.Button(button_frame, text="Очистить лог", command=self.clear_log)
        self.btn_clear_log.pack(side=tk.LEFT, padx=5)

        # Логи
        log_frame = ttk.LabelFrame(main_frame, text="Лог выполнения", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        self.root.after(100, self.check_queue)

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        self.log("Лог очищен")

    def load_models(self):
        def _load_models():
            try:
                self.det_model = YOLO("weights/det.pt", verbose=False)
                self.seg_model = YOLO("weights/seg.pt", verbose=False)
                self.molscribe = MolScribe("molscribe/swin_base_char_aux_1m.pth", device=self.device)

                self.message_queue.put(("log", "Все модели успешно загружены!"))

            except Exception as e:
                self.message_queue.put(("error", f"Ошибка загрузки моделей: {str(e)}"))

        threading.Thread(target=_load_models, daemon=True).start()

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images and PDF", "*.png *.jpg *.jpeg *.pdf")]
        )
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=f"Загружено: {os.path.basename(file_path)}")
            self.log(f"Загружен файл: {file_path}")

            if file_path.lower().endswith('.pdf'):
                self.log("Обнаружен PDF документ, будет выполнена обработка всех страниц")
            else:
                self.log("Обнаружено изображение, будет выполнена обработка одного файла")

    def select_output_dir(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir = dir_path
            self.output_label.config(text=f"Папка сохранения: {dir_path}")
            self.log(f"Выбрана папка для сохранения: {dir_path}")

    def start_processing(self):
        if not hasattr(self, 'file_path') or not hasattr(self, 'output_dir'):
            messagebox.showwarning("Ошибка", "Загрузите файл и выберите папку для сохранения!")
            return

        self.processing = True
        self.start_time = time.time()
        self.btn_process.config(state=tk.DISABLED)
        self.btn_load.config(state=tk.DISABLED)
        self.btn_output.config(state=tk.DISABLED)
        self.molscribe_check.config(state=tk.DISABLED)
        self.btn_clear_log.config(state=tk.DISABLED)

        threading.Thread(target=self.process_file, daemon=True).start()

    def check_queue(self):
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()
                if msg_type == "log":
                    self.log(msg_data)
                elif msg_type == "error":
                    messagebox.showerror("Ошибка", msg_data)
                    self.reset_ui()
                elif msg_type == "done":
                    messagebox.showinfo("Готово", msg_data)
                    self.reset_ui()

        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.update()

    def reset_ui(self):
        self.start_time = None
        self.processing = False
        self.btn_process.config(state=tk.NORMAL)
        self.btn_load.config(state=tk.NORMAL)
        self.btn_output.config(state=tk.NORMAL)
        self.molscribe_check.config(state=tk.NORMAL)
        self.btn_clear_log.config(state=tk.NORMAL)

    def process_file(self):
        try:
            file_start_time = time.time()

            if self.file_path.lower().endswith('.pdf'):
                self.process_pdf_file()
            else:
                self.process_single_image()

            file_elapsed = time.time() - file_start_time
            mins, secs = divmod(file_elapsed, 60)
            self.message_queue.put(("done", f"Все операции успешно завершены!\nВремя обработки: {int(mins)} мин {int(secs)} сек"))
        except Exception as e:
            self.message_queue.put(("error", f"Ошибка при обработке: {str(e)}"))
            self.message_queue.put(("log", f"Ошибка: {str(e)}"))

    def process_pdf_file(self):
        self.log(f"Начало обработки PDF документа: {self.file_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_path(
                self.file_path,
                output_folder=temp_dir,
                fmt='png',
                thread_count=4
            )

            self.log(f"PDF конвертирован в {len(images)} страниц(ы)")

            for i, image in enumerate(images):
                page_num = i + 1
                self.log(f"\nОбработка страницы {page_num} из {len(images)}")

                page_dir = os.path.join(self.output_dir, f"page_{page_num}")
                os.makedirs(page_dir, exist_ok=True)

                page_image_path = os.path.join(page_dir, "original.png")
                image.save(page_image_path, 'PNG')

                self.process_image(page_image_path, page_dir)

    def process_single_image(self):
        self.log(f"Начало обработки изображения: {self.file_path}")
        image_dir = os.path.join(self.output_dir, "image_results")
        os.makedirs(image_dir, exist_ok=True)

        original_path = os.path.join(image_dir, "original.png")
        img = cv2.imread(self.file_path)
        cv2.imwrite(original_path, img)

        self.process_image(self.file_path, image_dir)

    def process_image(self, image_path, output_dir):
        os.makedirs(os.path.join(output_dir, "structures"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reactions"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "structures_from_reactions"), exist_ok=True)

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        vis_img = img.copy()

        self.log("Запуск детекции структур и реакций...")
        det_results = self.det_model.predict(
            source=img,
            imgsz=1024,
            conf=0.5,
            device=self.device
        )
        self.log(f"Обнаружено {len(det_results[0].boxes)} объектов")

        for box in det_results[0].boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls)
            color = (0, 0, 255) if class_id == 1 else (255, 0, 0)  # Красный для структур, синий для реакций
            label = "Structure" if class_id == 1 else "Reaction"

            cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(vis_img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        reaction_counter = 1
        for i, box in enumerate(det_results[0].boxes):
            if int(box.cls) == 0:  # Реакции
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                reaction_img = img[ymin:ymax, xmin:xmax]

                self.log(f"Обработка реакции {reaction_counter}...")
                seg_results = self.seg_model.predict(
                    source=reaction_img,
                    device=self.device
                )

                mask = np.zeros_like(reaction_img)
                for seg_box in seg_results[0].boxes:
                    sxmin, symin, sxmax, symax = map(int, seg_box.xyxy[0].tolist())
                    class_id = int(seg_box.cls)

                    if class_id == 0:  # arrow
                        color = (0, 255, 0)  # Зеленый
                    elif class_id == 1:  # structure
                        color = (0, 0, 255)  # Красный
                    elif class_id == 2:  # txt
                        color = (0, 255, 255)  # Желтый

                    mask[symin:symax, sxmin:sxmax] = color

                alpha = 0.8
                reaction_img_vis = reaction_img.copy()
                reaction_img_vis = cv2.addWeighted(reaction_img_vis, 1, mask, alpha, 0)

                vis_img[ymin:ymax, xmin:xmax] = reaction_img_vis

                reaction_path = os.path.join(output_dir, "reactions", f"reaction{reaction_counter}.png")
                cv2.imwrite(reaction_path, reaction_img)
                self.log(f"Реакция {reaction_counter} сохранена: {reaction_path}")

                self.process_reaction_segmentation(seg_results, reaction_img, reaction_counter, output_dir)
                reaction_counter += 1

        vis_path = os.path.join(output_dir, "visualized_detection.png")
        cv2.imwrite(vis_path, vis_img)
        self.log(f"Визуализация сохранена: {vis_path}")

        structure_counter = 1
        for i, box in enumerate(det_results[0].boxes):
            if int(box.cls) == 1:  # Структуры
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                crop = img[ymin:ymax, xmin:xmax]

                structure_dir = os.path.join(output_dir, "structures", f"structure{structure_counter}")
                os.makedirs(structure_dir, exist_ok=True)
                cv2.imwrite(os.path.join(structure_dir, "structure.png"), crop)
                self.log(f"Структура {structure_counter} сохранена в {structure_dir}")

                if self.use_molscribe.get() and self.molscribe is not None:
                    self.process_with_molscribe(crop, structure_dir)
                structure_counter += 1

        self.log("Обработка изображения завершена успешно!")

    def process_reaction_segmentation(self, seg_results, reaction_img, reaction_idx, output_dir):
        structure_counter = 1
        for box in seg_results[0].boxes:
            if int(box.cls) == 1:  # Структуры внутри реакций
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                crop = reaction_img[ymin:ymax, xmin:xmax]

                dir_name = f"structure_r{reaction_idx}_{structure_counter}"
                structure_dir = os.path.join(output_dir, "structures_from_reactions", dir_name)
                os.makedirs(structure_dir, exist_ok=True)
                cv2.imwrite(os.path.join(structure_dir, "structure.png"), crop)
                self.log(f"Структура {structure_counter} из реакции {reaction_idx} сохранена в {structure_dir}")

                if self.use_molscribe.get() and self.molscribe is not None:
                    self.process_with_molscribe(crop, structure_dir)
                structure_counter += 1

    def process_with_molscribe(self, image, output_dir):
        temp_path = os.path.join(output_dir, "temp.png")
        cv2.imwrite(temp_path, image)

        try:
            self.log(f"Распознавание структуры в {output_dir}...")
            prediction = self.molscribe.predict_image_file(temp_path, return_atoms_bonds=True)

            with open(os.path.join(output_dir, "smiles.txt"), "w") as f:
                f.write(prediction['smiles'])

            self.log(f"Успешно распознана структура: {prediction['smiles']}")
        except Exception as e:
            self.log(f"Ошибка при распознавании структуры: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChemApp(root)
    root.mainloop()