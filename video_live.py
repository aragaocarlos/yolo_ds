import cv2
import os
import random
import numpy as np
import time
import threading
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import datetime

# Classe que representa cada objeto rastreado
class Track:
    def __init__(self, track_id, bbox):
        self.track_id = track_id  # ID único do objeto rastreado
        self.bbox = bbox  # Bounding box (x1, y1, x2, y2) do objeto

# Wrapper para simplificar o uso do DeepSORT
class DeepSortWrapper:
    def __init__(self, model_filename='model_data/mars-small128.pb', max_cosine_distance=0.4, nn_budget=None):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)  # Métrica de similaridade
        self.tracker = DeepSortTracker(metric)  # Inicializa o tracker com a métrica definida
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)  # Cria o codificador MARS
        self.tracks = []  # Lista de objetos rastreados

    # Atualiza o tracker com as detecções do frame atual
    def update(self, frame, detections):
        if len(detections) == 0:
            self.tracker.predict()  # Prediz posições futuras quando não há detecções
            self.tracker.update([])  # Atualiza tracker sem novas detecções
            self._update_tracks()  # Atualiza lista de tracks confirmadas
            return

        # Converte detecções para formato esperado (x1, y1, x2, y2)
        bboxes = np.array([d[:4] for d in detections])
        scores = [d[4] for d in detections]  # Extrai scores de confiança
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]  # Converte para formato width/height
        features = self.encoder(frame, bboxes)  # Gera embeddings MARS para cada bounding box

        # Cria objetos Detection combinando bbox, score e embeddings
        dets = [Detection(bboxes[i], scores[i], features[i]) for i in range(len(bboxes))]
        self.tracker.predict()  # Prediz posição futura de cada track
        self.tracker.update(dets)  # Atualiza tracker com as novas detecções
        self._update_tracks()  # Atualiza lista de tracks confirmadas

    # Atualiza a lista de tracks confirmadas
    def _update_tracks(self):
        active_tracks = []  # Lista temporária de tracks ativas
        for track in self.tracker.tracks:
            # Ignora tracks não confirmadas ou que não foram atualizadas recentemente
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()  # Converte bounding box para (top, left, bottom, right)
            active_tracks.append(Track(track.track_id, bbox))  # Adiciona track confirmada
        self.tracks = active_tracks  # Atualiza lista de tracks do wrapper

# Inicializa modelo YOLO e DeepSORT
model = YOLO("best.pt")  # Carrega o modelo YOLO treinado
deepsort = DeepSortWrapper(model_filename='mars-small128.pb')  # Inicializa DeepSORT com MARS

# Inicializa câmera
cap = cv2.VideoCapture()  # Cria objeto de captura de vídeo
ip = None  # Variável que armazenará IP da câmera (se necessário)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduz buffer para frames recentes

# Configurações de vídeo
frame_width, frame_height = 640, 360  # Resolução do frame
fps = 20  # Frames por segundo

# Armazenamento do último frame capturado
latest_frame = None  # Variável global para o frame mais recente
frame_lock = threading.Lock()  # Lock para acesso seguro à variável latest_frame entre threads

# Função que lê frames da câmera continuamente em uma thread
def camera_reader():
    global latest_frame
    while True:
        ret, frame = cap.read()  # Lê frame da câmera
        if not ret or frame is None:  # Verifica se a leitura falhou
            print("[WARN] Falha na leitura da câmera. Verifique o IP ou conexão.")
            time.sleep(0.5)
            continue
        try:
            frame = cv2.resize(frame, (frame_width, frame_height))  # Redimensiona para a resolução desejada
            with frame_lock:
                latest_frame = frame.copy()  # Armazena cópia do frame
        except Exception as e:
            print(f"[ERROR] Erro ao processar o frame: {e}")

# Inicia thread de leitura da câmera
camera_thread = threading.Thread(target=camera_reader, daemon=True)
camera_thread.start()

# Inicializa interface Tkinter
root = tk.Tk()
root.update_idletasks()  # Atualiza interface para medir tamanho da tela

# Cria o Frame principal que irá conter a interface
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)  # Padding nas laterais e em cima/embaixo

# Define o tamanho da janela e a centraliza
width = 1000
height = 720
x = (root.winfo_screenwidth() // 2) - (width // 2)
y = (root.winfo_screenheight() // 2) - (height // 2)
root.geometry(f"{width}x{height}+{x}+{y}")
root.resizable(True, True)  # Permite redimensionar janela

# Função para ajustar o tamanho da janela dependendo da câmera
def ajustar_tamanho():
    if ip:  # Se a câmera estiver conectada
        root.geometry(f"{frame_width + 100}x{frame_height + 300}")  # Expande a janela
    else:  # Se a câmera não estiver conectada
        root.geometry(f"{frame_width}x{frame_height}")  # Tamanho menor

ajustar_tamanho()  # Ajusta janela inicialmente
root.title("SIMBIC Rastreamento de Garrafas de Plástico")  # Define título da janela
video_frame = tk.Label(root)  # Label onde o vídeo será exibido
video_frame.pack()  # Posiciona label

# Variáveis de controle
is_tracking = False  # Indica se rastreamento está ativo
confidence_threshold = tk.DoubleVar(value=0.6)  # Limiar de confiança inicial
tracked_ids = set()  # Conjunto de IDs rastreados
colors = {}  # Dicionário para cores de cada ID
random.seed(42)  # Semente para cores aleatórias
frame_count = 0  # Contador de frames para FPS
start_time = time.time()  # Marca inicial para cálculo de FPS

# Arquivo de saída
output_file = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"  # Nome baseado na data/hora
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para salvar vídeo
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))  # Inicializa gravador
if not out.isOpened():
    print(f"[ERROR] Falha ao inicializar o gravador de vídeo para {output_file}")
else:
    print(f"[INFO] Gravando em: {output_file}")

# Função para iniciar/parar rastreamento
def toggle_tracking():
    global is_tracking
    is_tracking = not is_tracking  # Alterna estado
    tracking_button.config(text="Parar Rastreamento" if is_tracking else "Iniciar Rastreamento")  # Atualiza botão

    if is_tracking:
        confidence_slider.config(state=tk.DISABLED)  # Bloqueia slider
    else:
        confidence_slider.config(state=tk.NORMAL)  # Libera slider
        out.release()  # Salva vídeo
        messagebox.showinfo("Vídeo Salvo", f"O vídeo foi salvo com sucesso como:\n{output_file}")

# Converte frame para imagem Tkinter e atualiza interface
def show_frame(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte BGR para RGB
    img_pil = Image.fromarray(img_rgb)  # Cria objeto PIL
    img_tk = ImageTk.PhotoImage(img_pil)  # Converte para Tkinter
    video_frame.img_tk = img_tk  # Armazena referência para evitar garbage collection
    video_frame.config(image=img_tk)  # Atualiza label

# Função principal de processamento de cada frame
def process_frame():
    global frame_count, start_time

    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None  # Captura frame atual

    if frame is None:
        root.after(1, process_frame)  # Reagenda função
        return
    else:
        tracking_button.config(state=tk.NORMAL)  # Ativa botão quando frame disponível

    if is_tracking:
        results = model(frame, verbose=False)  # Executa detecção YOLO
        detections = []

        # Percorre cada detecção retornada pelo YOLO
        for result in results:
            for det in result.boxes.data:
                x1, y1, x2, y2, score, class_id = det.tolist()
                if score < confidence_threshold.get():  # Filtra por limiar
                    continue
                detections.append([x1, y1, x2, y2, score])  # Adiciona detecção válida

        deepsort.update(frame, detections)  # Atualiza tracker com novas detecções

        # Desenha bounding boxes e IDs rastreados
        for track in deepsort.tracks:
            x1, y1, x2, y2 = map(int, track.bbox)  # Converte bbox para inteiro
            tid = track.track_id  # Obtém ID do track
            tracked_ids.add(tid)  # Adiciona ao conjunto

            # Define cor aleatória se ID ainda não tiver
            if tid not in colors:
                colors[tid] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            color = colors[tid]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Desenha bbox
            cv2.putText(frame, f"ID: {tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  # Escreve ID

        total_ids = len(tracked_ids)  # Total de objetos rastreados
        resultado = total_ids * 2  # Resultado calculado

        # Texto a ser exibido
        texto1 = f"Total: {total_ids}"
        texto2 = f"Resultado: {resultado}"

        # Fonte e estilo
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        escala = 0.5
        espessura = 2

        # Tamanho dos textos
        (tw1, th1), _ = cv2.getTextSize(texto1, fonte, escala, espessura)
        (tw2, th2), _ = cv2.getTextSize(texto2, fonte, escala, espessura)

        # Posição dos retângulos
        x, y = 20, 20
        padding = 10
        altura_total = th1 + th2 + padding * 3
        largura_max = max(tw1, tw2) + padding * 2

        # Desenha fundo retangular para os textos
        cv2.rectangle(frame, (x, y), (x + largura_max, y + altura_total), (0, 0, 139), -1)

        # Escreve textos em branco sobre o fundo
        cv2.putText(frame, texto1, (x + padding, y + th1 + padding), fonte, escala, (255, 255, 255), espessura)
        cv2.putText(frame, texto2, (x + padding, y + th1 + th2 + padding * 2), fonte, escala, (255, 255, 255), espessura)

        out.write(frame)  # Salva frame no vídeo

        # Cálculo de FPS
        frame_count += 1
        if frame_count >= 10:
            end_time = time.time()
            print(f"[INFO] FPS: {frame_count / (end_time - start_time):.2f}")
            frame_count = 0
            start_time = time.time()

    show_frame(frame)  # Atualiza frame na interface
    root.after(1, process_frame)  # Reagenda processamento

# Widgets para IP da câmera
ip_label = tk.Label(root, text="Endereço IP da Câmera:")
ip_label.pack(pady=10)

ip_frame = tk.Frame(root)
ip_frame.pack(pady=10)

ip_entry = tk.Entry(ip_frame, width=40)
ip_entry.insert(0, "192.168.0.107:8080")  # IP inicial
ip_entry.pack(side=tk.LEFT, padx=10)

# Conecta câmera IP
def conectar_camera():
    global ip, cap
    ip_raw = ip_entry.get().strip()  # Captura IP inserido
    if not ip_raw.startswith("http"):
        ip = f"https://{ip_raw}/video"  # Formata URL
    else:
        ip = ip_raw
    cap.open(ip, cv2.CAP_FFMPEG)  # Tenta abrir stream
    if not cap.isOpened():
        print("[ERRO] Falha ao conectar com o IP informado.")
    else:
        print(f"[INFO] Conectado à câmera IP: {ip}")
        ajustar_tamanho()  # Ajusta tamanho da janela

conectar_button = tk.Button(ip_frame, text="Conectar Câmera", command=conectar_camera)
conectar_button.pack(side=tk.LEFT)

# Slider de confiança
confidence_label = tk.Label(root, text="Limiar de confiança")
confidence_label.pack(anchor="center")

confidence_slider = tk.Scale(
    root, from_=0.1, to=1.0, resolution=0.01,
    orient=tk.HORIZONTAL, variable=confidence_threshold, length=300
)
confidence_slider.pack(pady=5)

# Botão de rastreamento
tracking_button = tk.Button(root, text="Iniciar Rastreamento", command=toggle_tracking)
tracking_button.pack(pady=20)
tracking_button.config(state=tk.DISABLED)  # Desabilitado até frame disponível

# Início do loop de processamento
root.after(1, process_frame)
root.mainloop()  # Inicia interface

# Libera recursos
cap.release()
out.release()
cv2.destroyAllWindows()
