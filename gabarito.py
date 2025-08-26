import cv2
import numpy as np

# -----------------------
# Funções utilitárias
# -----------------------

def preprocess_image(img):
    """Pré-processa a imagem capturada da câmera."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def detect_bubbles(thresh):
    """Detecta contornos circulares (bolhas)."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for c in contours:
        area = cv2.contourArea(c)
        # Ajusta a faixa de área para ignorar contornos pequenos
        # como os de letras ou números das questões.
        if 400 < area < 2000:  
            (x,y,w,h) = cv2.boundingRect(c)
            ratio = w / float(h)
            # Aumenta a rigidez para contornos circulares
            # para evitar a detecção de formas alongadas.
            if 0.9 <= ratio <= 1.1:  
                bubbles.append((x,y,w,h,c))
    return bubbles

def group_bubbles(bubbles):
    """Agrupa bolhas por linha de questão."""
    bubbles = sorted(bubbles, key=lambda b: b[1])
    questions = []
    line = []
    last_y = None
    tol = 20
    for b in bubbles:
        x,y,w,h,c = b
        if last_y is None or abs(y - last_y) < tol:
            line.append(b)
            last_y = y
        else:
            questions.append(sorted(line, key=lambda bb: bb[0]))
            line = [b]
            last_y = y
    if line:
        questions.append(sorted(line, key=lambda bb: bb[0]))
    return questions

def get_marked_alternatives(questions, thresh):
    """Identifica alternativa marcada."""
    answers = []
    for q in questions:
        intensities = []
        for (x,y,w,h,c) in q:
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mean = cv2.mean(thresh, mask=mask)[0]
            intensities.append(mean)
        marked = np.argmax(intensities) # Mudei de np.argmin para np.argmax
        answers.append(marked)
    return answers

def evaluate_and_draw(frame, questions, answers, gabarito):
    """Compara respostas com gabarito e desenha no frame."""
    options = ["A","B","C","D","E"]
    acertos = 0

    for i, ans in enumerate(answers, start=1):
        marcado = options[ans] if ans < len(options) else "?"
        certo = gabarito.get(i, "?")

        cor = (0,0,255)  # vermelho = errado
        if marcado == certo:
            cor = (0,255,0)  # verde = certo
            acertos += 1

        # destacar a bolha escolhida
        x,y,w,h,c = questions[i-1][ans]
        cv2.rectangle(frame, (x,y), (x+w,y+h), cor, 2)
        cv2.putText(frame, f"Q{i}:{marcado}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

    # mostrar total no canto da tela
    cv2.putText(frame, f"Acertos: {acertos}/{len(answers)}", (30,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    cv2.putText(frame, f"Acertos: {acertos}/{len(answers)}", (30,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    return frame

# -----------------------
# Programa com webcam
# -----------------------

def main():
    # Gabarito esperado (da sua folha)
    gabarito = {
    1:"B", 2:"C", 3:"E", 4:"E", 5:"C",
    }


    cap = cv2.VideoCapture(0)  # abre webcam padrão
    if not cap.isOpened():
        print("Erro: câmera não encontrada")
        return

    # Tenta definir uma resolução maior (por exemplo, 1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Pressione 'c' para capturar e corrigir, 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define a área de interesse (ROI) para o scanner
        # Você deve ajustar esses valores para a sua câmera
        x = 400
        y = 50
        w = 500
        h = 600

        # Desenha o retângulo na janela para o usuário ver
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Camera - Pressione 'c' para capturar", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):  # capturar e corrigir
            # Recorta a imagem para a área de interesse
            roi = frame[y:y+h, x:x+w]

            # Processa apenas a área recortada (ROI)
            thresh = preprocess_image(roi)
            bubbles = detect_bubbles(thresh)
            questions = group_bubbles(bubbles)
            answers = get_marked_alternatives(questions, thresh)

            corrected = roi.copy()
            corrected = evaluate_and_draw(corrected, questions, answers, gabarito)

            cv2.imshow("Resultado", corrected)
            # A janela de resultado fica pausada até o usuário pressionar qualquer tecla
            cv2.waitKey(0)

        elif key == ord("q"):  # sair
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
