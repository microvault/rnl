import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, TextBox, Button
from matplotlib.patches import Rectangle
import json

class BBoxAnnotator:
    def __init__(self, image_path):
        self.image_path = image_path
        self.bboxes = []         # lista de {"label": str, "bbox": [x1, y1, x2, y2]}
        self.current_coords = [] # último bbox desenhado pelo RectangleSelector

        # Carrega a imagem
        self.fig, self.ax = plt.subplots()
        self.image = plt.imread(self.image_path)
        self.ax.imshow(self.image)
        # Deixa espaço no rodapé pros widgets
        plt.subplots_adjust(bottom=0.2)

        # RectangleSelector pra desenhar bounding box (fica fixo ao soltar)
        self.selector = RectangleSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=False
        )

        # Caixa de texto pro label
        ax_text = plt.axes([0.1, 0.05, 0.3, 0.05])
        self.text_box = TextBox(ax_text, 'Label: ')
        self.text_box.on_submit(self.on_text_submit)

        # Botão "Adicionar" - fixa o label naquele bbox
        ax_button_add = plt.axes([0.42, 0.05, 0.15, 0.05])
        self.btn_add = Button(ax_button_add, 'Adicionar')
        self.btn_add.on_clicked(self.add_bbox)

        # Botão "Salvar"
        ax_button_save = plt.axes([0.6, 0.05, 0.15, 0.05])
        self.btn_save = Button(ax_button_save, 'Salvar')
        self.btn_save.on_clicked(self.save_annotations)

        plt.show()

    def on_select(self, eclick, erelease):
        """
        Callback chamado quando arrasta e solta o mouse.
        Desenha o retângulo permanente assim que o mouse é solto.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        # Ordena para que x1 < x2 e y1 < y2
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        # Guarda as coords do bbox
        self.current_coords = [x1, y1, x2, y2]

        # Desenha retângulo permanente
        rect = Rectangle(
            (x1, y1), (x2 - x1), (y2 - y1),
            fill=False, edgecolor='blue', linewidth=2
        )
        self.ax.add_patch(rect)

        # Atualiza a figura
        self.fig.canvas.draw_idle()

    def on_text_submit(self, text):
        """
        Callback ao pressionar Enter na caixa de texto.
        Não precisa fazer nada extra aqui, pois vamos usar o botão "Adicionar".
        """
        pass

    def add_bbox(self, event):
        """
        Ao clicar em 'Adicionar', associamos o label ao bbox atual
        e desenhamos o label na imagem de modo permanente.
        """
        if not self.current_coords:
            print("Nenhum bounding box desenhado.")
            return

        label = self.text_box.text.strip()
        if not label:
            print("Label vazio, digite algo na TextBox.")
            return

        # Salva bbox e label
        self.bboxes.append({
            "label": label,
            "bbox": self.current_coords
        })

        # Desenha o label na figura
        x1, y1, x2, y2 = self.current_coords
        self.ax.text(
            x1, y1, label,
            color='blue',
            fontsize=9,
            backgroundcolor='white'
        )

        # Limpa a TextBox e as coords
        self.text_box.set_val('')
        self.current_coords = []

        plt.draw()

    def save_annotations(self, event):
        """
        Salva as anotações em JSON e a figura anotada em PNG.
        """
        # Salva JSON
        with open("annotations.json", "w") as f:
            json.dump(self.bboxes, f, indent=2)

        # Salva imagem final com bounding boxes e labels
        self.fig.savefig("annotated_image.png")
        print("Anotações salvas em 'annotations.json' e 'annotated_image.png'.")

# Exemplo de uso
if __name__ == "__main__":
    BBoxAnnotator("/Users/nicolasalan/microvault/rnl/contour.png")
