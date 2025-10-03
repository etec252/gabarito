import cv2
import numpy as np

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 150, 225, cv2.THRESH_BINARY_INV)[1]
    return thresh

def detect_bubbles(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for c in contours:
        area = cv2.contourArea(c)
        if 300 < area < 3000:
            x, y, w, h = cv2.boundingRect(c)
            ratio = w / float(h)
            if 0.8 <= ratio <= 1.2:
                bubbles.append((x, y, w, h, c))
    return bubbles

def group_bubbles_by_columns(bubbles, num_columns=4):
    if not bubbles:
        return []

    bubbles = sorted(bubbles, key=lambda b: b[0])
    min_x, max_x = min([b[0] for b in bubbles]), max([b[0] for b in bubbles])
    col_width = (max_x - min_x) / num_columns

    columns = [[] for _ in range(num_columns)]
    for b in bubbles:
        x = b[0]
        col_index = min(int((x - min_x) / (col_width + 5)), num_columns - 1)
        columns[col_index].append(b)

    all_questions = []
    for col in columns:
        col_sorted = sorted(col, key=lambda b: b[1])
        line = []
        last_y = None
        tol = 25
        for b in col_sorted:
            x, y, w, h, c = b
            if last_y is None or abs(y - last_y) < tol:
                line.append(b)
                last_y = y
            else:
                all_questions.append(sorted(line, key=lambda bb: bb[0]))
                line = [b]
                last_y = y
        if line:
            all_questions.append(sorted(line, key=lambda bb: bb[0]))

    return all_questions

def get_marked_alternatives(questions, thresh):
    answers = []
    for q in questions:
        if len(q) != 5:
            answers.append(None)
            continue
        intensities = []
        for (x, y, w, h, c) in q:
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mean = cv2.mean(thresh, mask=mask)[0]
            intensities.append(mean)
        marked = np.argmax(intensities)
        answers.append(marked)
    return answers

def avaliar_e_desenhar(imagem, questions, answers, gabarito):
    options = ["A", "B", "C", "D", "E"]
    acertos = 0

    for i, ans in enumerate(answers):
        if ans is None or i + 1 not in gabarito:
            continue

        marcado = options[ans]
        correto = gabarito[i + 1]
        cor = (0, 0, 255)  # vermelho = errado
        if marcado == correto:
            cor = (0, 255, 0)  # verde = certo
            acertos += 1

        x, y, w, h, c = questions[i][ans]
        cv2.rectangle(imagem, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(imagem, f"{i+1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)

    cv2.putText(imagem, f"Acertos: {acertos}/{len(gabarito)}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.putText(imagem, f"Acertos: {acertos}/{len(gabarito)}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return imagem

def main():
    gabarito = {
        1: "C", 2: "D", 3: "D", 4: "C", 5: "E", 6: "C", 7: "B", 8: "C", 9: "B", 10: "E",
        11: "C", 12: "D", 13: "B", 14: "A", 15: "A", 16: "D", 17: "E", 18: "C", 19: "D", 20: "C",
        21: "B", 22: "A", 23: "C", 24: "E", 25: "D", 26: "D", 27: "B", 28: "B", 29: "E", 30: "D",
        31: "A", 32: "D", 33: "D", 34: "C", 35: "D", 36: "A", 37: "A", 38: "E", 39: "B", 40: "E",
        41: "B", 42: "E", 43: "C", 44: "C", 45: "C", 46: "C", 47: "E", 48: "C", 49: "B", 50: "E",
        51: "E", 52: "A", 53: "A", 54: "E", 55: "D", 56: "E", 57: "C", 58: "C", 59: "A", 60: "B",
        61: "E", 62: "B", 63: "D", 64: "E", 65: "E", 66: "A", 67: "D", 68: "B", 69: "C", 70: "C"
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: câmera não encontrada")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Pressione 'c' para capturar e corrigir, 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define ROI
        x, y, w, h = 300, 50, 650, 700
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Scanner - pressione 'c' para capturar", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            roi = frame[y:y + h, x:x + w]
            roi = cv2.resize(roi, (1000, int(roi.shape[0] * 1000 / roi.shape[1])))

            thresh = preprocess_image(roi)
            bubbles = detect_bubbles(thresh)
            questions = group_bubbles_by_columns(bubbles)
            if not questions:
                print("Nenhuma bolha detectada.")
                continue

            answers = get_marked_alternatives(questions, thresh)
            resultado = avaliar_e_desenhar(roi.copy(), questions, answers, gabarito)

            cv2.imshow("Resultado", resultado)
            cv2.waitKey(0)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
