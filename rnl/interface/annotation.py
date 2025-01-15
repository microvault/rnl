import cv2
import json
from numba import njit

@njit
def calc_area(x1, y1, x2, y2):
    return abs((x2 - x1) * (y2 - y1))

drawing = False
ix = iy = 0
rects_info = []

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, rects_info

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        # Cria um retângulo vazio
        rects_info.append({'label': '', 'x1': x, 'y1': y, 'x2': x, 'y2': y, 'area': 0})

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Atualiza as coordenadas do último retângulo
        rects_info[-1]['x2'] = x
        rects_info[-1]['y2'] = y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rects_info[-1]['x2'] = x
        rects_info[-1]['y2'] = y
        area = calc_area(ix, iy, x, y)
        rects_info[-1]['area'] = area
        # Pede um nome pra label
        rects_info[-1]['label'] = input("Nome da label: ")

def annotate_png(png_path, json_path="labels.json"):
    global rects_info
    rects_info = []

    # Carrega imagem PNG
    img = cv2.imread(png_path)
    clone = img.copy()

    cv2.namedWindow("PNG Annotation")
    cv2.setMouseCallback("PNG Annotation", mouse_callback)

    while True:
        temp = clone.copy()
        # Desenha todos os retângulos criados
        for r in rects_info:
            cv2.rectangle(temp, (r['x1'], r['y1']), (r['x2'], r['y2']), (0, 255, 0), 2)
            cv2.putText(temp, r['label'], (r['x1'], r['y1'] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("PNG Annotation", temp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC pra sair
            break

    cv2.destroyAllWindows()

    # Salva dados em JSON
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(rects_info, f, ensure_ascii=False, indent=2)

# Exemplo de uso:
annotate_png("/Users/nicolasalan/microvault/rnl/contour_mask.png", "labels.json")
